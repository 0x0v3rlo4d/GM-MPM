/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

/** \file
 * \ingroup freestyle
 * \brief Class to define a Bezier curve of order 4.
 */

#include <vector>

#include "Geom.h"

#include "../system/FreestyleConfig.h"

#include "MEM_guardedalloc.h"

namespace Freestyle {

using namespace Geometry;

class BezierCurveSegment {
 private:
  std::vector<Vec2d> _ControlPolygon;
  std::vector<Vec2d> _Vertices;

 public:
  virtual ~BezierCurveSegment();

  void AddControlPoint(const Vec2d &iPoint);
  void Build();

  inline int size() const
  {
    return _ControlPolygon.size();
  }

  inline std::vector<Vec2d> &vertices()
  {
    return _Vertices;
  }

  MEM_CXX_CLASS_ALLOC_FUNCS("Freestyle:BezierCurveSegment")
};

class BezierCurve {
 private:
  std::vector<Vec2d> _ControlPolygon;
  std::vector<BezierCurveSegment *> _Segments;
  BezierCurveSegment *_currentSegment;

 public:
  BezierCurve();
  BezierCurve(std::vector<Vec2d> &iPoints, double error = 4.0);
  virtual ~BezierCurve();

  void AddControlPoint(const Vec2d &iPoint);

  std::vector<Vec2d> &controlPolygon()
  {
    return _ControlPolygon;
  }

  std::vector<BezierCurveSegment *> &segments()
  {
    return _Segments;
  }

  MEM_CXX_CLASS_ALLOC_FUNCS("Freestyle:BezierCurve")
};

} /* namespace Freestyle */
