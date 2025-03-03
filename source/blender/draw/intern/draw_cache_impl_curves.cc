/* SPDX-FileCopyrightText: 2017 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw
 *
 * \brief Curves API for render engines
 */

#include <cstring>

#include "MEM_guardedalloc.h"

#include "BLI_array_utils.hh"
#include "BLI_listbase.h"
#include "BLI_math_base.h"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_string.h"
#include "BLI_task.hh"
#include "BLI_utildefines.h"

#include "DNA_curves_types.h"
#include "DNA_object_types.h"
#include "DNA_userdef_types.h"

#include "DEG_depsgraph_query.hh"

#include "BKE_crazyspace.hh"
#include "BKE_curves.hh"
#include "BKE_curves_utils.hh"
#include "BKE_customdata.hh"
#include "BKE_geometry_set.hh"

#include "GPU_batch.hh"
#include "GPU_context.hh"
#include "GPU_material.hh"
#include "GPU_texture.hh"

#include "DRW_render.hh"

#include "draw_attributes.hh"
#include "draw_cache_impl.hh" /* own include */
#include "draw_cache_inline.hh"
#include "draw_curves_private.hh" /* own include */

namespace blender::draw {

#define EDIT_CURVES_NURBS_CONTROL_POINT (1u)
#define EDIT_CURVES_BEZIER_HANDLE (1u << 1)
#define EDIT_CURVES_ACTIVE_HANDLE (1u << 2)
/* Bezier curve control point lying on the curve.
 * The one between left and right handles. */
#define EDIT_CURVES_BEZIER_KNOT (1u << 3)
#define EDIT_CURVES_HANDLE_TYPES_SHIFT (4u)

/* ---------------------------------------------------------------------- */

struct CurvesBatchCache {
  CurvesEvalCache eval_cache;

  gpu::Batch *edit_points;
  gpu::Batch *edit_handles;

  gpu::Batch *sculpt_cage;
  gpu::IndexBuf *sculpt_cage_ibo;

  /* Crazy-space point positions for original points. */
  gpu::VertBuf *edit_points_pos;

  /* Additional data needed for shader to choose color for each point in edit_points_pos.
   * If first bit is set, then point is NURBS control point. EDIT_CURVES_NURBS_CONTROL_POINT is
   * used to set and test. If second, then point is Bezier handle point. Set and tested with
   * EDIT_CURVES_BEZIER_HANDLE.
   * In Bezier case two handle types of HandleType are also encoded.
   * Byte structure for Bezier knot point (handle middle point):
   * | left handle type | right handle type |      | BEZIER|  NURBS|
   * | 7              6 | 5               4 | 3  2 |     1 |     0 |
   *
   * If it is left or right handle point, then same handle type is repeated in both slots.
   */
  gpu::VertBuf *edit_points_data;

  /* Selection of original points. */
  gpu::VertBuf *edit_points_selection;

  gpu::IndexBuf *edit_handles_ibo;

  gpu::Batch *edit_curves_lines;
  gpu::VertBuf *edit_curves_lines_pos;
  gpu::IndexBuf *edit_curves_lines_ibo;

  /* Whether the cache is invalid. */
  bool is_dirty;

  /**
   * The draw cache extraction is currently not multi-threaded for multiple objects, but if it was,
   * some locking would be necessary because multiple objects can use the same curves data with
   * different materials, etc. This is a placeholder to make multi-threading easier in the future.
   */
  std::mutex render_mutex;
};

static bool batch_cache_is_dirty(const Curves &curves)
{
  const CurvesBatchCache *cache = static_cast<CurvesBatchCache *>(curves.batch_cache);
  return (cache && cache->is_dirty == false);
}

static void init_batch_cache(Curves &curves)
{
  CurvesBatchCache *cache = static_cast<CurvesBatchCache *>(curves.batch_cache);

  if (!cache) {
    cache = MEM_new<CurvesBatchCache>(__func__);
    curves.batch_cache = cache;
  }
  else {
    cache->eval_cache = {};
  }

  cache->is_dirty = false;
}

static void discard_attributes(CurvesEvalCache &eval_cache)
{
  for (const int i : IndexRange(GPU_MAX_ATTR)) {
    GPU_VERTBUF_DISCARD_SAFE(eval_cache.proc_attributes_buf[i]);
  }

  for (const int j : IndexRange(GPU_MAX_ATTR)) {
    GPU_VERTBUF_DISCARD_SAFE(eval_cache.final.attributes_buf[j]);
  }

  drw_attributes_clear(&eval_cache.final.attr_used);
}

static void clear_edit_data(CurvesBatchCache *cache)
{
  /* TODO: more granular update tagging. */
  GPU_VERTBUF_DISCARD_SAFE(cache->edit_points_pos);
  GPU_VERTBUF_DISCARD_SAFE(cache->edit_points_data);
  GPU_VERTBUF_DISCARD_SAFE(cache->edit_points_selection);
  GPU_INDEXBUF_DISCARD_SAFE(cache->edit_handles_ibo);

  GPU_BATCH_DISCARD_SAFE(cache->edit_points);
  GPU_BATCH_DISCARD_SAFE(cache->edit_handles);

  GPU_INDEXBUF_DISCARD_SAFE(cache->sculpt_cage_ibo);
  GPU_BATCH_DISCARD_SAFE(cache->sculpt_cage);

  GPU_VERTBUF_DISCARD_SAFE(cache->edit_curves_lines_pos);
  GPU_INDEXBUF_DISCARD_SAFE(cache->edit_curves_lines_ibo);
  GPU_BATCH_DISCARD_SAFE(cache->edit_curves_lines);
}

static void clear_final_data(CurvesEvalFinalCache &final_cache)
{
  GPU_VERTBUF_DISCARD_SAFE(final_cache.proc_buf);
  GPU_BATCH_DISCARD_SAFE(final_cache.proc_hairs);
  for (const int j : IndexRange(GPU_MAX_ATTR)) {
    GPU_VERTBUF_DISCARD_SAFE(final_cache.attributes_buf[j]);
  }
}

static void clear_eval_data(CurvesEvalCache &eval_cache)
{
  /* TODO: more granular update tagging. */
  GPU_VERTBUF_DISCARD_SAFE(eval_cache.proc_point_buf);
  GPU_VERTBUF_DISCARD_SAFE(eval_cache.proc_length_buf);
  GPU_VERTBUF_DISCARD_SAFE(eval_cache.proc_strand_buf);
  GPU_VERTBUF_DISCARD_SAFE(eval_cache.proc_strand_seg_buf);

  clear_final_data(eval_cache.final);

  discard_attributes(eval_cache);
}

static void clear_batch_cache(Curves &curves)
{
  CurvesBatchCache *cache = static_cast<CurvesBatchCache *>(curves.batch_cache);
  if (!cache) {
    return;
  }

  clear_eval_data(cache->eval_cache);
  clear_edit_data(cache);
}

static CurvesBatchCache &get_batch_cache(Curves &curves)
{
  DRW_curves_batch_cache_validate(&curves);
  return *static_cast<CurvesBatchCache *>(curves.batch_cache);
}

struct PositionAndParameter {
  float3 position;
  float parameter;
};

static void fill_points_position_time_vbo(const OffsetIndices<int> points_by_curve,
                                          const Span<float3> positions,
                                          MutableSpan<PositionAndParameter> posTime_data,
                                          MutableSpan<float> hairLength_data)
{
  threading::parallel_for(points_by_curve.index_range(), 1024, [&](const IndexRange range) {
    for (const int i_curve : range) {
      const IndexRange points = points_by_curve[i_curve];

      Span<float3> curve_positions = positions.slice(points);
      MutableSpan<PositionAndParameter> curve_posTime_data = posTime_data.slice(points);

      float total_len = 0.0f;
      for (const int i_point : curve_positions.index_range()) {
        if (i_point > 0) {
          total_len += math::distance(curve_positions[i_point - 1], curve_positions[i_point]);
        }
        curve_posTime_data[i_point].position = curve_positions[i_point];
        curve_posTime_data[i_point].parameter = total_len;
      }
      hairLength_data[i_curve] = total_len;

      /* Assign length value. */
      if (total_len > 0.0f) {
        const float factor = 1.0f / total_len;
        /* Divide by total length to have a [0-1] number. */
        for (const int i_point : curve_positions.index_range()) {
          curve_posTime_data[i_point].parameter *= factor;
        }
      }
    }
  });
}

static void create_points_position_time_vbo(const bke::CurvesGeometry &curves,
                                            CurvesEvalCache &cache)
{
  GPUVertFormat format = {0};
  GPU_vertformat_attr_add(&format, "posTime", GPU_COMP_F32, 4, GPU_FETCH_FLOAT);

  cache.proc_point_buf = GPU_vertbuf_create_with_format_ex(
      format, GPU_USAGE_STATIC | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);
  GPU_vertbuf_data_alloc(*cache.proc_point_buf, cache.points_num);

  GPUVertFormat length_format = {0};
  GPU_vertformat_attr_add(&length_format, "hairLength", GPU_COMP_F32, 1, GPU_FETCH_FLOAT);

  cache.proc_length_buf = GPU_vertbuf_create_with_format_ex(
      length_format, GPU_USAGE_STATIC | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);
  GPU_vertbuf_data_alloc(*cache.proc_length_buf, cache.curves_num);

  /* TODO: Only create hairLength VBO when necessary. */
  fill_points_position_time_vbo(curves.points_by_curve(),
                                curves.positions(),
                                cache.proc_point_buf->data<PositionAndParameter>(),
                                cache.proc_length_buf->data<float>());
}

static uint32_t bezier_data_value(int8_t handle_type, bool is_active)
{
  return (handle_type << EDIT_CURVES_HANDLE_TYPES_SHIFT) | EDIT_CURVES_BEZIER_HANDLE |
         (is_active ? EDIT_CURVES_ACTIVE_HANDLE : 0);
}

static void create_edit_points_position_and_data(
    const bke::CurvesGeometry &curves,
    const IndexMask bezier_curves,
    const OffsetIndices<int> bezier_dst_offsets,
    const bke::crazyspace::GeometryDeformation deformation,
    CurvesBatchCache &cache)
{
  static const GPUVertFormat format_pos = GPU_vertformat_from_attribute(
      "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);
  /* GPU_COMP_U32 is used instead of GPU_COMP_U8 because depending on running hardware stride might
   * still be 4. Thus adding complexity to the code and still sparing no memory. */
  static const GPUVertFormat format_data = GPU_vertformat_from_attribute(
      "data", GPU_COMP_U32, 1, GPU_FETCH_INT);

  Span<float3> deformed_positions = deformation.positions;
  const int bezier_point_count = bezier_dst_offsets.total_size();
  const int size = deformed_positions.size() + bezier_point_count * 2;
  GPU_vertbuf_init_with_format(*cache.edit_points_pos, format_pos);
  GPU_vertbuf_data_alloc(*cache.edit_points_pos, size);

  GPU_vertbuf_init_with_format(*cache.edit_points_data, format_data);
  GPU_vertbuf_data_alloc(*cache.edit_points_data, size);

  MutableSpan<float3> pos_dst = cache.edit_points_pos->data<float3>();
  pos_dst.take_front(deformed_positions.size()).copy_from(deformed_positions);

  MutableSpan<uint32_t> data_dst = cache.edit_points_data->data<uint32_t>();

  MutableSpan<uint32_t> handle_data_left(data_dst.data() + deformed_positions.size(),
                                         bezier_point_count);
  MutableSpan<uint32_t> handle_data_right(
      data_dst.data() + deformed_positions.size() + bezier_point_count, bezier_point_count);

  const Span<float3> left_handle_positions = curves.handle_positions_left();
  const Span<float3> right_handle_positions = curves.handle_positions_right();
  const VArray<int8_t> left_handle_types = curves.handle_types_left();
  const VArray<int8_t> right_handle_types = curves.handle_types_right();
  const OffsetIndices<int> points_by_curve = curves.points_by_curve();

  const VArray<bool> selection_attr = *curves.attributes().lookup_or_default<bool>(
      ".selection", bke::AttrDomain::Point, true);

  auto handle_other_curves = [&](const uint32_t fill_value, const bool mark_active) {
    return [&, fill_value, mark_active](const IndexMask &selection) {
      selection.foreach_index(GrainSize(256), [&](const int curve_i) {
        const IndexRange points = points_by_curve[curve_i];
        bool is_active = false;
        if (mark_active) {
          is_active = array_utils::count_booleans(selection_attr, points) > 0;
        }
        uint32_t data_value = fill_value | (is_active ? EDIT_CURVES_ACTIVE_HANDLE : 0u);
        data_dst.slice(points).fill(data_value);
      });
    };
  };

  bke::curves::foreach_curve_by_type(
      curves.curve_types(),
      curves.curve_type_counts(),
      curves.curves_range(),
      handle_other_curves(0, false),
      handle_other_curves(0, false),
      [&](const IndexMask &selection) {
        const VArray<bool> selection_left = *curves.attributes().lookup_or_default<bool>(
            ".selection_handle_left", bke::AttrDomain::Point, true);
        const VArray<bool> selection_right = *curves.attributes().lookup_or_default<bool>(
            ".selection_handle_right", bke::AttrDomain::Point, true);

        selection.foreach_index(GrainSize(256), [&](const int src_i, const int64_t dst_i) {
          for (const int point : points_by_curve[src_i]) {
            const int point_in_curve = point - points_by_curve[src_i].start();
            const int dst_index = bezier_dst_offsets[dst_i].start() + point_in_curve;

            data_dst[point] = EDIT_CURVES_BEZIER_KNOT;
            bool is_active = selection_attr[point] || selection_left[point] ||
                             selection_right[point];
            handle_data_left[dst_index] = bezier_data_value(left_handle_types[point], is_active);
            handle_data_right[dst_index] = bezier_data_value(right_handle_types[point], is_active);
          }
        });
      },
      handle_other_curves(EDIT_CURVES_NURBS_CONTROL_POINT, true));

  if (!bezier_point_count) {
    return;
  }

  MutableSpan<float3> left_handles(pos_dst.data() + deformed_positions.size(), bezier_point_count);
  MutableSpan<float3> right_handles(
      pos_dst.data() + deformed_positions.size() + bezier_point_count, bezier_point_count);

  /* TODO: Use deformed left_handle_positions and left_handle_positions. */
  array_utils::gather_group_to_group(
      points_by_curve, bezier_dst_offsets, bezier_curves, left_handle_positions, left_handles);
  array_utils::gather_group_to_group(
      points_by_curve, bezier_dst_offsets, bezier_curves, right_handle_positions, right_handles);
}

static void create_edit_points_selection(const bke::CurvesGeometry &curves,
                                         const IndexMask bezier_curves,
                                         const OffsetIndices<int> bezier_dst_offsets,
                                         CurvesBatchCache &cache)
{
  static const GPUVertFormat format_data = GPU_vertformat_from_attribute(
      "selection", GPU_COMP_F32, 1, GPU_FETCH_FLOAT);

  const int bezier_point_count = bezier_dst_offsets.total_size();
  const int vert_count = curves.points_num() + bezier_point_count * 2;
  GPU_vertbuf_init_with_format(*cache.edit_points_selection, format_data);
  GPU_vertbuf_data_alloc(*cache.edit_points_selection, vert_count);
  MutableSpan<float> data = cache.edit_points_selection->data<float>();

  const VArray<float> attribute = *curves.attributes().lookup_or_default<float>(
      ".selection", bke::AttrDomain::Point, 1.0f);
  attribute.materialize(data.slice(0, curves.points_num()));

  if (!bezier_point_count) {
    return;
  }

  const VArray<float> attribute_left = *curves.attributes().lookup_or_default<float>(
      ".selection_handle_left", bke::AttrDomain::Point, 1.0f);
  const VArray<float> attribute_right = *curves.attributes().lookup_or_default<float>(
      ".selection_handle_right", bke::AttrDomain::Point, 1.0f);

  const OffsetIndices<int> points_by_curve = curves.points_by_curve();

  IndexRange dst_range = IndexRange::from_begin_size(curves.points_num(), bezier_point_count);
  array_utils::gather_group_to_group(
      points_by_curve, bezier_dst_offsets, bezier_curves, attribute_left, data.slice(dst_range));

  dst_range = dst_range.shift(bezier_point_count);
  array_utils::gather_group_to_group(
      points_by_curve, bezier_dst_offsets, bezier_curves, attribute_right, data.slice(dst_range));
}

static void create_lines_ibo_no_cyclic(const OffsetIndices<int> points_by_curve,
                                       gpu::IndexBuf &ibo)
{
  const int points_num = points_by_curve.total_size();
  const int curves_num = points_by_curve.size();
  const int indices_num = points_num + curves_num;
  GPUIndexBufBuilder builder;
  GPU_indexbuf_init(&builder, GPU_PRIM_LINE_STRIP, indices_num, points_num);
  MutableSpan<uint> ibo_data = GPU_indexbuf_get_data(&builder);
  threading::parallel_for(IndexRange(curves_num), 1024, [&](const IndexRange range) {
    for (const int curve : range) {
      const IndexRange points = points_by_curve[curve];
      const IndexRange ibo_range = IndexRange(points.start() + curve, points.size() + 1);
      for (const int i : points.index_range()) {
        ibo_data[ibo_range[i]] = points[i];
      }
      ibo_data[ibo_range.last()] = gpu::RESTART_INDEX;
    }
  });
  GPU_indexbuf_build_in_place_ex(&builder, 0, points_num, true, &ibo);
}

static void create_lines_ibo_with_cyclic(const OffsetIndices<int> points_by_curve,
                                         const Span<bool> cyclic,
                                         gpu::IndexBuf &ibo)
{
  const int points_num = points_by_curve.total_size();
  const int curves_num = points_by_curve.size();
  const int indices_num = points_num + curves_num * 2;
  GPUIndexBufBuilder builder;
  GPU_indexbuf_init(&builder, GPU_PRIM_LINE_STRIP, indices_num, points_num);
  MutableSpan<uint> ibo_data = GPU_indexbuf_get_data(&builder);
  threading::parallel_for(IndexRange(curves_num), 1024, [&](const IndexRange range) {
    for (const int curve : range) {
      const IndexRange points = points_by_curve[curve];
      const IndexRange ibo_range = IndexRange(points.start() + curve * 2, points.size() + 2);
      for (const int i : points.index_range()) {
        ibo_data[ibo_range[i]] = points[i];
      }
      ibo_data[ibo_range.last(1)] = cyclic[curve] ? points.first() : gpu::RESTART_INDEX;
      ibo_data[ibo_range.last()] = gpu::RESTART_INDEX;
    }
  });
  GPU_indexbuf_build_in_place_ex(&builder, 0, points_num, true, &ibo);
}

static void create_lines_ibo_with_cyclic(const OffsetIndices<int> points_by_curve,
                                         const VArray<bool> &cyclic,
                                         gpu::IndexBuf &ibo)
{
  const array_utils::BooleanMix cyclic_mix = array_utils::booleans_mix_calc(cyclic);
  if (cyclic_mix == array_utils::BooleanMix::AllFalse) {
    create_lines_ibo_no_cyclic(points_by_curve, ibo);
  }
  else {
    const VArraySpan<bool> cyclic_span(cyclic);
    create_lines_ibo_with_cyclic(points_by_curve, cyclic_span, ibo);
  }
}

static void calc_edit_handles_ibo(const bke::CurvesGeometry &curves,
                                  const IndexMask bezier_curves,
                                  const OffsetIndices<int> bezier_offsets,
                                  const IndexMask other_curves,
                                  CurvesBatchCache &cache)
{
  const int bezier_point_count = bezier_offsets.total_size();
  /* Left and right handle will be appended for each Bezier point. */
  const int vert_len = curves.points_num() + 2 * bezier_point_count;
  /* For each point has 2 lines from 2 points. */
  const int index_len_for_bezier_handles = 4 * bezier_point_count;
  const VArray<bool> cyclic = curves.cyclic();
  /* For curves like NURBS each control point except last generates two point line.
   * If one point curves or two point cyclic curves are present, not all builder's buffer space
   * will be used. */
  const int index_len_for_other_handles = (curves.points_num() - bezier_point_count -
                                           other_curves.size()) *
                                              2 +
                                          array_utils::count_booleans(cyclic, other_curves) * 2;
  const int index_len = index_len_for_other_handles + index_len_for_bezier_handles;
  /* Use two index buffer builders for the same underlying memory. */
  GPUIndexBufBuilder elb, right_elb;
  GPU_indexbuf_init_ex(&elb, GPU_PRIM_LINES, index_len, vert_len);
  memcpy(&right_elb, &elb, sizeof(elb));
  right_elb.index_len = 2 * bezier_point_count;

  const OffsetIndices points_by_curve = curves.points_by_curve();

  bezier_curves.foreach_index([&](const int64_t src_i, const int64_t dst_i) {
    IndexRange bezier_points = points_by_curve[src_i];
    const int index_shift = curves.points_num() - bezier_points.first() +
                            bezier_offsets[dst_i].first();
    for (const int point : bezier_points) {
      const int point_left_i = index_shift + point;
      GPU_indexbuf_add_line_verts(&elb, point_left_i, point);
      GPU_indexbuf_add_line_verts(&right_elb, point_left_i + bezier_point_count, point);
    }
  });
  other_curves.foreach_index([&](const int64_t src_i) {
    IndexRange curve_points = points_by_curve[src_i];
    if (curve_points.size() <= 1) {
      return;
    }
    for (const int point : curve_points.drop_back(1)) {
      GPU_indexbuf_add_line_verts(&right_elb, point, point + 1);
    }
    if (cyclic[src_i] && curve_points.size() > 2) {
      GPU_indexbuf_add_line_verts(&right_elb, curve_points.first(), curve_points.last());
    }
  });
  GPU_indexbuf_join(&elb, &right_elb);
  GPU_indexbuf_build_in_place(&elb, cache.edit_handles_ibo);
}

static void alloc_final_attribute_vbo(CurvesEvalCache &cache,
                                      const GPUVertFormat &format,
                                      const int index,
                                      const char * /*name*/)
{
  cache.final.attributes_buf[index] = GPU_vertbuf_create_with_format_ex(
      format, GPU_USAGE_DEVICE_ONLY | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);

  /* Create a destination buffer for the transform feedback. Sized appropriately */
  /* Those are points! not line segments. */
  GPU_vertbuf_data_alloc(*cache.final.attributes_buf[index],
                         cache.final.resolution * cache.curves_num);
}

static void ensure_control_point_attribute(const Curves &curves,
                                           CurvesEvalCache &cache,
                                           const DRW_AttributeRequest &request,
                                           const int index,
                                           const GPUVertFormat &format)
{
  if (cache.proc_attributes_buf[index] != nullptr) {
    return;
  }

  GPU_VERTBUF_DISCARD_SAFE(cache.proc_attributes_buf[index]);

  cache.proc_attributes_buf[index] = GPU_vertbuf_create_with_format_ex(
      format, GPU_USAGE_STATIC | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);
  gpu::VertBuf &attr_vbo = *cache.proc_attributes_buf[index];

  GPU_vertbuf_data_alloc(attr_vbo,
                         request.domain == bke::AttrDomain::Point ? curves.geometry.point_num :
                                                                    curves.geometry.curve_num);

  const bke::AttributeAccessor attributes = curves.geometry.wrap().attributes();

  /* TODO(@kevindietrich): float4 is used for scalar attributes as the implicit conversion done
   * by OpenGL to vec4 for a scalar `s` will produce a `vec4(s, 0, 0, 1)`. However, following
   * the Blender convention, it should be `vec4(s, s, s, 1)`. This could be resolved using a
   * similar texture state swizzle to map the attribute correctly as for volume attributes, so we
   * can control the conversion ourselves. */
  bke::AttributeReader<ColorGeometry4f> attribute = attributes.lookup_or_default<ColorGeometry4f>(
      request.attribute_name, request.domain, {0.0f, 0.0f, 0.0f, 1.0f});

  MutableSpan<ColorGeometry4f> vbo_span = attr_vbo.data<ColorGeometry4f>();

  attribute.varray.materialize(vbo_span);
}

static void ensure_final_attribute(const Curves &curves,
                                   CurvesEvalCache &cache,
                                   const DRW_AttributeRequest &request,
                                   const int index)
{
  char sampler_name[32];
  drw_curves_get_attribute_sampler_name(request.attribute_name, sampler_name);

  GPUVertFormat format = {0};
  /* All attributes use vec4, see comment below. */
  GPU_vertformat_attr_add(&format, sampler_name, GPU_COMP_F32, 4, GPU_FETCH_FLOAT);

  ensure_control_point_attribute(curves, cache, request, index, format);

  /* Existing final data may have been for a different attribute (with a different name or domain),
   * free the data. */
  GPU_VERTBUF_DISCARD_SAFE(cache.final.attributes_buf[index]);

  /* Ensure final data for points. */
  if (request.domain == bke::AttrDomain::Point) {
    alloc_final_attribute_vbo(cache, format, index, sampler_name);
  }
}

static void fill_curve_offsets_vbos(const OffsetIndices<int> points_by_curve,
                                    GPUVertBufRaw &data_step,
                                    GPUVertBufRaw &seg_step)
{
  for (const int i : points_by_curve.index_range()) {
    const IndexRange points = points_by_curve[i];

    *(uint *)GPU_vertbuf_raw_step(&data_step) = points.start();
    *(ushort *)GPU_vertbuf_raw_step(&seg_step) = points.size() - 1;
  }
}

static void create_curve_offsets_vbos(const OffsetIndices<int> points_by_curve,
                                      CurvesEvalCache &cache)
{
  GPUVertBufRaw data_step, seg_step;

  GPUVertFormat format_data = {0};
  uint data_id = GPU_vertformat_attr_add(&format_data, "data", GPU_COMP_U32, 1, GPU_FETCH_INT);

  GPUVertFormat format_seg = {0};
  uint seg_id = GPU_vertformat_attr_add(&format_seg, "data", GPU_COMP_U16, 1, GPU_FETCH_INT);

  /* Curve Data. */
  cache.proc_strand_buf = GPU_vertbuf_create_with_format_ex(
      format_data, GPU_USAGE_STATIC | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);
  GPU_vertbuf_data_alloc(*cache.proc_strand_buf, cache.curves_num);
  GPU_vertbuf_attr_get_raw_data(cache.proc_strand_buf, data_id, &data_step);

  cache.proc_strand_seg_buf = GPU_vertbuf_create_with_format_ex(
      format_seg, GPU_USAGE_STATIC | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);
  GPU_vertbuf_data_alloc(*cache.proc_strand_seg_buf, cache.curves_num);
  GPU_vertbuf_attr_get_raw_data(cache.proc_strand_seg_buf, seg_id, &seg_step);

  fill_curve_offsets_vbos(points_by_curve, data_step, seg_step);
}

static void alloc_final_points_vbo(CurvesEvalCache &cache)
{
  /* Same format as proc_point_buf. */
  GPUVertFormat format = {0};
  GPU_vertformat_attr_add(&format, "pos", GPU_COMP_F32, 4, GPU_FETCH_FLOAT);

  cache.final.proc_buf = GPU_vertbuf_create_with_format_ex(
      format, GPU_USAGE_DEVICE_ONLY | GPU_USAGE_FLAG_BUFFER_TEXTURE_ONLY);

  /* Create a destination buffer for the transform feedback. Sized appropriately */

  /* Those are points! not line segments. */
  uint point_len = cache.final.resolution * cache.curves_num;
  /* Avoid creating null sized VBO which can lead to crashes on certain platforms. */
  point_len = max_ii(1, point_len);

  GPU_vertbuf_data_alloc(*cache.final.proc_buf, point_len);
}

static void calc_final_indices(const bke::CurvesGeometry &curves,
                               CurvesEvalCache &cache,
                               const int thickness_res)
{
  BLI_assert(thickness_res <= MAX_THICKRES); /* Cylinder strip not currently supported. */
  /* Determine prim type and element count.
   * NOTE: Metal backend uses non-restart prim types for optimal HW performance. */
  bool use_strip_prims = (GPU_backend_get_type() != GPU_BACKEND_METAL);
  int verts_per_curve;
  GPUPrimType prim_type;

  if (use_strip_prims) {
    /* +1 for primitive restart */
    verts_per_curve = cache.final.resolution * thickness_res;
    prim_type = (thickness_res == 1) ? GPU_PRIM_LINE_STRIP : GPU_PRIM_TRI_STRIP;
  }
  else {
    /* Use full primitive type. */
    prim_type = (thickness_res == 1) ? GPU_PRIM_LINES : GPU_PRIM_TRIS;
    int verts_per_segment = ((prim_type == GPU_PRIM_LINES) ? 2 : 6);
    verts_per_curve = (cache.final.resolution - 1) * verts_per_segment;
  }

  static const GPUVertFormat format = GPU_vertformat_from_attribute(
      "dummy", GPU_COMP_U32, 1, GPU_FETCH_INT_TO_FLOAT_UNIT);

  gpu::VertBuf *vbo = GPU_vertbuf_create_with_format(format);
  GPU_vertbuf_data_alloc(*vbo, 1);

  gpu::IndexBuf *ibo = nullptr;
  eGPUBatchFlag owns_flag = GPU_BATCH_OWNS_VBO;
  if (curves.curves_num()) {
    ibo = GPU_indexbuf_build_curves_on_device(prim_type, curves.curves_num(), verts_per_curve);
    owns_flag |= GPU_BATCH_OWNS_INDEX;
  }
  cache.final.proc_hairs = GPU_batch_create_ex(prim_type, vbo, ibo, owns_flag);
}

static bool ensure_attributes(const Curves &curves,
                              CurvesBatchCache &cache,
                              const GPUMaterial *gpu_material)
{
  const CustomData &cd_curve = curves.geometry.curve_data;
  const CustomData &cd_point = curves.geometry.point_data;
  CurvesEvalFinalCache &final_cache = cache.eval_cache.final;

  if (gpu_material) {
    /* The following code should be kept in sync with `mesh_cd_calc_used_gpu_layers`. */
    DRW_Attributes attrs_needed;
    drw_attributes_clear(&attrs_needed);
    ListBase gpu_attrs = GPU_material_attributes(gpu_material);
    LISTBASE_FOREACH (const GPUMaterialAttribute *, gpu_attr, &gpu_attrs) {
      const char *name = gpu_attr->name;
      eCustomDataType type = static_cast<eCustomDataType>(gpu_attr->type);
      int layer = -1;
      std::optional<bke::AttrDomain> domain;

      if (gpu_attr->type == CD_AUTO_FROM_NAME) {
        /* We need to deduce what exact layer is used.
         *
         * We do it based on the specified name.
         */
        if (name[0] != '\0') {
          layer = CustomData_get_named_layer(&cd_curve, CD_PROP_FLOAT2, name);
          type = CD_MTFACE;
          domain = bke::AttrDomain::Curve;

          if (layer == -1) {
            /* Try to match a generic attribute, we use the first attribute domain with a
             * matching name. */
            if (drw_custom_data_match_attribute(cd_point, name, &layer, &type)) {
              domain = bke::AttrDomain::Point;
            }
            else if (drw_custom_data_match_attribute(cd_curve, name, &layer, &type)) {
              domain = bke::AttrDomain::Curve;
            }
            else {
              domain.reset();
              layer = -1;
            }
          }

          if (layer == -1) {
            continue;
          }
        }
        else {
          /* Fall back to the UV layer, which matches old behavior. */
          type = CD_MTFACE;
        }
      }
      else {
        if (drw_custom_data_match_attribute(cd_curve, name, &layer, &type)) {
          domain = bke::AttrDomain::Curve;
        }
        else if (drw_custom_data_match_attribute(cd_point, name, &layer, &type)) {
          domain = bke::AttrDomain::Point;
        }
      }

      switch (type) {
        case CD_MTFACE: {
          if (layer == -1) {
            layer = (name[0] != '\0') ?
                        CustomData_get_named_layer(&cd_curve, CD_PROP_FLOAT2, name) :
                        CustomData_get_render_layer(&cd_curve, CD_PROP_FLOAT2);
            if (layer != -1) {
              domain = bke::AttrDomain::Curve;
            }
          }
          if (layer == -1) {
            layer = (name[0] != '\0') ?
                        CustomData_get_named_layer(&cd_point, CD_PROP_FLOAT2, name) :
                        CustomData_get_render_layer(&cd_point, CD_PROP_FLOAT2);
            if (layer != -1) {
              domain = bke::AttrDomain::Point;
            }
          }

          if (layer != -1 && name[0] == '\0' && domain.has_value()) {
            name = CustomData_get_layer_name(
                domain == bke::AttrDomain::Curve ? &cd_curve : &cd_point, CD_PROP_FLOAT2, layer);
          }

          if (layer != -1 && domain.has_value()) {
            drw_attributes_add_request(&attrs_needed, name, CD_PROP_FLOAT2, layer, *domain);
          }
          break;
        }

        case CD_TANGENT:
        case CD_ORCO:
          break;

        case CD_PROP_BYTE_COLOR:
        case CD_PROP_COLOR:
        case CD_PROP_QUATERNION:
        case CD_PROP_FLOAT3:
        case CD_PROP_BOOL:
        case CD_PROP_INT8:
        case CD_PROP_INT32:
        case CD_PROP_INT16_2D:
        case CD_PROP_INT32_2D:
        case CD_PROP_FLOAT:
        case CD_PROP_FLOAT2: {
          if (layer != -1 && domain.has_value()) {
            drw_attributes_add_request(&attrs_needed, name, type, layer, *domain);
          }
          break;
        }
        default:
          break;
      }
    }

    if (!drw_attributes_overlap(&final_cache.attr_used, &attrs_needed)) {
      /* Some new attributes have been added, free all and start over. */
      for (const int i : IndexRange(GPU_MAX_ATTR)) {
        GPU_VERTBUF_DISCARD_SAFE(final_cache.attributes_buf[i]);
        GPU_VERTBUF_DISCARD_SAFE(cache.eval_cache.proc_attributes_buf[i]);
      }
      drw_attributes_merge(&final_cache.attr_used, &attrs_needed, cache.render_mutex);
    }
    drw_attributes_merge(&final_cache.attr_used_over_time, &attrs_needed, cache.render_mutex);
  }

  bool need_tf_update = false;

  for (const int i : IndexRange(final_cache.attr_used.num_requests)) {
    const DRW_AttributeRequest &request = final_cache.attr_used.requests[i];

    if (cache.eval_cache.final.attributes_buf[i] != nullptr) {
      continue;
    }

    if (request.domain == bke::AttrDomain::Point) {
      need_tf_update = true;
    }

    ensure_final_attribute(curves, cache.eval_cache, request, i);
  }

  return need_tf_update;
}

static void request_attribute(Curves &curves, const char *name)
{
  CurvesBatchCache &cache = get_batch_cache(curves);
  CurvesEvalFinalCache &final_cache = cache.eval_cache.final;

  DRW_Attributes attributes{};

  bke::CurvesGeometry &curves_geometry = curves.geometry.wrap();
  std::optional<bke::AttributeMetaData> meta_data = curves_geometry.attributes().lookup_meta_data(
      name);
  if (!meta_data) {
    return;
  }
  const bke::AttrDomain domain = meta_data->domain;
  const eCustomDataType type = meta_data->data_type;
  const CustomData &custom_data = domain == bke::AttrDomain::Point ? curves.geometry.point_data :
                                                                     curves.geometry.curve_data;

  drw_attributes_add_request(
      &attributes, name, type, CustomData_get_named_layer(&custom_data, type, name), domain);

  drw_attributes_merge(&final_cache.attr_used, &attributes, cache.render_mutex);
}

void drw_curves_get_attribute_sampler_name(const char *layer_name, char r_sampler_name[32])
{
  char attr_safe_name[GPU_MAX_SAFE_ATTR_NAME];
  GPU_vertformat_safe_attr_name(layer_name, attr_safe_name, GPU_MAX_SAFE_ATTR_NAME);
  /* Attributes use auto-name. */
  BLI_snprintf(r_sampler_name, 32, "a%s", attr_safe_name);
}

bool curves_ensure_procedural_data(Curves *curves_id,
                                   CurvesEvalCache **r_cache,
                                   const GPUMaterial *gpu_material,
                                   const int subdiv,
                                   const int thickness_res)
{
  const bke::CurvesGeometry &curves = curves_id->geometry.wrap();
  bool need_ft_update = false;

  CurvesBatchCache &cache = get_batch_cache(*curves_id);
  CurvesEvalCache &eval_cache = cache.eval_cache;

  if (eval_cache.final.hair_subdiv != subdiv || eval_cache.final.thickres != thickness_res) {
    /* If the subdivision or indexing settings have changed, the evaluation cache is cleared. */
    clear_final_data(eval_cache.final);
    eval_cache.final.hair_subdiv = subdiv;
    eval_cache.final.thickres = thickness_res;
  }

  eval_cache.curves_num = curves.curves_num();
  eval_cache.points_num = curves.points_num();

  const int steps = 3; /* TODO: don't hard-code? */
  eval_cache.final.resolution = 1 << (steps + subdiv);

  /* Refreshed on combing and simulation. */
  if (eval_cache.proc_point_buf == nullptr || DRW_vbo_requested(eval_cache.proc_point_buf)) {
    create_points_position_time_vbo(curves, eval_cache);
    need_ft_update = true;
  }

  /* Refreshed if active layer or custom data changes. */
  if (eval_cache.proc_strand_buf == nullptr) {
    create_curve_offsets_vbos(curves.points_by_curve(), eval_cache);
  }

  /* Refreshed only on subdiv count change. */
  if (eval_cache.final.proc_buf == nullptr) {
    alloc_final_points_vbo(eval_cache);
    need_ft_update = true;
  }

  if (eval_cache.final.proc_hairs == nullptr) {
    calc_final_indices(curves, eval_cache, thickness_res);
  }
  eval_cache.final.thickres = thickness_res;

  need_ft_update |= ensure_attributes(*curves_id, cache, gpu_material);

  *r_cache = &eval_cache;
  return need_ft_update;
}

void DRW_curves_batch_cache_dirty_tag(Curves *curves, int mode)
{
  CurvesBatchCache *cache = static_cast<CurvesBatchCache *>(curves->batch_cache);
  if (cache == nullptr) {
    return;
  }
  switch (mode) {
    case BKE_CURVES_BATCH_DIRTY_ALL:
      cache->is_dirty = true;
      break;
    default:
      BLI_assert_unreachable();
  }
}

void DRW_curves_batch_cache_validate(Curves *curves)
{
  if (!batch_cache_is_dirty(*curves)) {
    clear_batch_cache(*curves);
    init_batch_cache(*curves);
  }
}

void DRW_curves_batch_cache_free(Curves *curves)
{
  clear_batch_cache(*curves);
  CurvesBatchCache *batch_cache = static_cast<CurvesBatchCache *>(curves->batch_cache);
  MEM_delete(batch_cache);
  curves->batch_cache = nullptr;
}

void DRW_curves_batch_cache_free_old(Curves *curves, int ctime)
{
  CurvesBatchCache *cache = static_cast<CurvesBatchCache *>(curves->batch_cache);
  if (cache == nullptr) {
    return;
  }

  bool do_discard = false;

  CurvesEvalFinalCache &final_cache = cache->eval_cache.final;

  if (drw_attributes_overlap(&final_cache.attr_used_over_time, &final_cache.attr_used)) {
    final_cache.last_attr_matching_time = ctime;
  }

  if (ctime - final_cache.last_attr_matching_time > U.vbotimeout) {
    do_discard = true;
  }

  drw_attributes_clear(&final_cache.attr_used_over_time);

  if (do_discard) {
    discard_attributes(cache->eval_cache);
  }
}

gpu::Batch *DRW_curves_batch_cache_get_edit_points(Curves *curves)
{
  CurvesBatchCache &cache = get_batch_cache(*curves);
  return DRW_batch_request(&cache.edit_points);
}

gpu::Batch *DRW_curves_batch_cache_get_sculpt_curves_cage(Curves *curves)
{
  CurvesBatchCache &cache = get_batch_cache(*curves);
  return DRW_batch_request(&cache.sculpt_cage);
}

gpu::Batch *DRW_curves_batch_cache_get_edit_curves_handles(Curves *curves)
{
  CurvesBatchCache &cache = get_batch_cache(*curves);
  return DRW_batch_request(&cache.edit_handles);
}

gpu::Batch *DRW_curves_batch_cache_get_edit_curves_lines(Curves *curves)
{
  CurvesBatchCache &cache = get_batch_cache(*curves);
  return DRW_batch_request(&cache.edit_curves_lines);
}

gpu::VertBuf **DRW_curves_texture_for_evaluated_attribute(Curves *curves,
                                                          const char *name,
                                                          bool *r_is_point_domain)
{
  CurvesBatchCache &cache = get_batch_cache(*curves);
  CurvesEvalFinalCache &final_cache = cache.eval_cache.final;

  request_attribute(*curves, name);

  int request_i = -1;
  for (const int i : IndexRange(final_cache.attr_used.num_requests)) {
    if (STREQ(final_cache.attr_used.requests[i].attribute_name, name)) {
      request_i = i;
      break;
    }
  }
  if (request_i == -1) {
    *r_is_point_domain = false;
    return nullptr;
  }
  switch (final_cache.attr_used.requests[request_i].domain) {
    case bke::AttrDomain::Point:
      *r_is_point_domain = true;
      return &final_cache.attributes_buf[request_i];
    case bke::AttrDomain::Curve:
      *r_is_point_domain = false;
      return &cache.eval_cache.proc_attributes_buf[request_i];
    default:
      BLI_assert_unreachable();
      return nullptr;
  }
}

static void create_edit_points_position_vbo(
    const bke::CurvesGeometry &curves,
    const bke::crazyspace::GeometryDeformation & /*deformation*/,
    CurvesBatchCache &cache)
{
  static const GPUVertFormat format = GPU_vertformat_from_attribute(
      "pos", GPU_COMP_F32, 3, GPU_FETCH_FLOAT);

  /* TODO: Deform curves using deformations. */
  const Span<float3> positions = curves.evaluated_positions();
  GPU_vertbuf_init_with_format(*cache.edit_curves_lines_pos, format);
  GPU_vertbuf_data_alloc(*cache.edit_curves_lines_pos, positions.size());
  cache.edit_curves_lines_pos->data<float3>().copy_from(positions);
}

void DRW_curves_batch_cache_create_requested(Object *ob)
{
  Curves *curves_id = static_cast<Curves *>(ob->data);
  Object *ob_orig = DEG_get_original_object(ob);
  if (ob_orig == nullptr) {
    return;
  }
  const Curves *curves_orig_id = static_cast<Curves *>(ob_orig->data);

  draw::CurvesBatchCache &cache = draw::get_batch_cache(*curves_id);
  const bke::CurvesGeometry &curves_orig = curves_orig_id->geometry.wrap();

  bool is_edit_data_needed = false;

  if (DRW_batch_requested(cache.edit_points, GPU_PRIM_POINTS)) {
    DRW_vbo_request(cache.edit_points, &cache.edit_points_pos);
    DRW_vbo_request(cache.edit_points, &cache.edit_points_data);
    DRW_vbo_request(cache.edit_points, &cache.edit_points_selection);
    is_edit_data_needed = true;
  }
  if (DRW_batch_requested(cache.sculpt_cage, GPU_PRIM_LINE_STRIP)) {
    DRW_ibo_request(cache.sculpt_cage, &cache.sculpt_cage_ibo);
    DRW_vbo_request(cache.sculpt_cage, &cache.edit_points_pos);
    DRW_vbo_request(cache.sculpt_cage, &cache.edit_points_data);
    DRW_vbo_request(cache.sculpt_cage, &cache.edit_points_selection);
    is_edit_data_needed = true;
  }
  if (DRW_batch_requested(cache.edit_handles, GPU_PRIM_LINES)) {
    DRW_ibo_request(cache.edit_handles, &cache.edit_handles_ibo);
    DRW_vbo_request(cache.edit_handles, &cache.edit_points_pos);
    DRW_vbo_request(cache.edit_handles, &cache.edit_points_data);
    DRW_vbo_request(cache.edit_handles, &cache.edit_points_selection);
    is_edit_data_needed = true;
  }
  if (DRW_batch_requested(cache.edit_curves_lines, GPU_PRIM_LINE_STRIP)) {
    DRW_vbo_request(cache.edit_curves_lines, &cache.edit_curves_lines_pos);
    DRW_ibo_request(cache.edit_curves_lines, &cache.edit_curves_lines_ibo);
  }

  const bke::crazyspace::GeometryDeformation deformation =
      is_edit_data_needed || DRW_vbo_requested(cache.edit_curves_lines_pos) ?
          bke::crazyspace::get_evaluated_curves_deformation(ob, *ob_orig) :
          bke::crazyspace::GeometryDeformation();

  if (DRW_ibo_requested(cache.sculpt_cage_ibo)) {
    create_lines_ibo_no_cyclic(curves_orig.points_by_curve(), *cache.sculpt_cage_ibo);
  }

  if (DRW_vbo_requested(cache.edit_curves_lines_pos)) {
    create_edit_points_position_vbo(curves_orig, deformation, cache);
  }

  if (DRW_ibo_requested(cache.edit_curves_lines_ibo)) {
    create_lines_ibo_with_cyclic(curves_orig.evaluated_points_by_curve(),
                                 curves_orig.cyclic(),
                                 *cache.edit_curves_lines_ibo);
  }

  if (!is_edit_data_needed) {
    return;
  }

  IndexMaskMemory memory;
  const IndexMask bezier_curves = bke::curves::indices_for_type(curves_orig.curve_types(),
                                                                curves_orig.curve_type_counts(),
                                                                CURVE_TYPE_BEZIER,
                                                                curves_orig.curves_range(),
                                                                memory);
  Array<int> bezier_point_offset_data(bezier_curves.size() + 1);
  const OffsetIndices<int> bezier_offsets = offset_indices::gather_selected_offsets(
      curves_orig.points_by_curve(), bezier_curves, bezier_point_offset_data);

  if (DRW_vbo_requested(cache.edit_points_pos)) {
    create_edit_points_position_and_data(
        curves_orig, bezier_curves, bezier_offsets, deformation, cache);
  }
  if (DRW_vbo_requested(cache.edit_points_selection)) {
    create_edit_points_selection(curves_orig, bezier_curves, bezier_offsets, cache);
  }
  if (DRW_ibo_requested(cache.edit_handles_ibo)) {
    const IndexMask other_curves = bezier_curves.complement(curves_orig.curves_range(), memory);
    calc_edit_handles_ibo(curves_orig, bezier_curves, bezier_offsets, other_curves, cache);
  }
}

}  // namespace blender::draw
