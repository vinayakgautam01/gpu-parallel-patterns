#pragma once

namespace gpp::stencil {

/// 7-point 3D stencil weights.
///
///   out[i] = c*center + xn*(-x) + xp*(+x)
///          + yn*(-y)  + yp*(+y)
///          + zn*(-z)  + zp*(+z)
///
/// Layout: row-major with Z as the outermost dimension.
///   index(x,y,z) = z * (nx * ny) + y * nx + x
struct Weights7 {
    float c;    // center
    float xn;   // x-1
    float xp;   // x+1
    float yn;   // y-1
    float yp;   // y+1
    float zn;   // z-1
    float zp;   // z+1
};

}  // namespace gpp::stencil
