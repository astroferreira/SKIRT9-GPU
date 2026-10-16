// Exact orthonormal detector basis from SKIRT/core/FrameInstrument.cpp.
export const MAX_FOV_KPC = 10000;
export const MAX_TARGET_KPC = 10000;
export function cameraBasis({ inclination, azimuth, roll }) {
  const [t, p, w] = [inclination, azimuth, roll].map(x => x * Math.PI / 180);
  const [ct, st, cp, sp, cw, sw] = [Math.cos(t), Math.sin(t), Math.cos(p), Math.sin(p), Math.cos(w), Math.sin(w)];
  return {
    right: [-cw*sp + sw*cp*ct, cw*cp + sw*sp*ct, -sw*st],
    up: [-sw*sp - cw*cp*ct, sw*cp - cw*sp*ct, cw*st],
    observer: [st*cp, st*sp, ct],
  };
}
export const defaults = fov => ({ inclination: 55, azimuth: 35, roll: 0, target_kpc: [0,0,0], fov_kpc: fov });
export function preset(name, fov) {
  const views = {xy:[0,0,90], xz:[90,-90,0], yz:[90,0,0], oblique:[55,35,0]};
  const [inclination,azimuth,roll] = views[name];
  return {...defaults(fov), inclination,azimuth,roll};
}
export const wrapAngle = x => ((x + 180) % 360 + 360) % 360 - 180;
