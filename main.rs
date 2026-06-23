//! swiss_cheese.rs — a procedurally carved cheese wheel for Bevy 0.19.
//!
//! Cargo.toml:
//!   [dependencies]
//!   bevy = "0.19"
//!
//! Run with `cargo run --release` (debug-built marching is slow at high res).
//!
//! The pipeline is: define an implicit field (signed distance, negative = "inside
//! cheese"), sample it on a grid, and extract the zero-isosurface as triangles.
//! There is deliberately no constructive-solid-geometry step on meshes — every
//! "subtraction" is just a max() on the field, which is why this stays robust
//! exactly where mesh-boolean CSG falls apart (spheres straddling the flat cuts).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
// NOTE: if a future point release relocates PrimitiveTopology, it also lives at
// bevy::render::render_resource::PrimitiveTopology. Mesh/Indices are in bevy_mesh
// (re-exported here as bevy::mesh) since 0.17.
use bevy::prelude::*;

// ---------------------------------------------------------------------------
// The spec: everything that makes one wheel distinct. Defaults give a tidy,
// camera-ready wheel, so callers only override what they care about.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct CheeseSpec {
    pub wheel_radius: f32,
    pub wheel_height: f32,
    /// Angular size of the *removed* slice, in radians, in the half-open range
    /// [0, TAU). The surviving cheese spans `TAU - wedge_bite`, so:
    ///   * an untouched round:          wedge_bite = 0.0         (kept = 360°)
    ///   * a small nibble of a wheel:   wedge_bite = FRAC_PI_3   (kept ≈ 300°)
    ///   * a half-disk:                 wedge_bite = PI          (kept = 180°)
    ///   * a quarter wedge of cheese:   wedge_bite = TAU - FRAC_PI_2  (kept = 90°)
    ///   * a near-invisible sliver:     wedge_bite -> TAU        (kept -> 0°)
    /// The slice is symmetric about the z = 0 plane and the field is sign-exact
    /// across the whole range — the old convex-only (< PI) restriction is gone.
    /// A bite of exactly 0 is special-cased in `cheese_field` to a clean round
    /// (the naive formula would otherwise carve a zero-width slit at z = 0).
    pub wedge_bite: f32,
    pub hole_count: usize,
    /// (min, max) bubble radius. Inclusive range, sampled per hole.
    pub hole_radius: (f32, f32),
    /// Determinism knob. Same seed -> byte-identical holes, no RNG involved.
    /// Defaulting it to `wedge_bite` honours "placement by radians cut as seed":
    /// the size of the bite literally decides where the bubbles land.
    pub hole_seed: f32,
}

impl Default for CheeseSpec {
    fn default() -> Self {
        // A half-disk bite (180°). NOTE: this is PI, not FRAC_PI_3 — the previous
        // "60-degree" comment disagreed with the value it sat next to. Value left
        // unchanged so existing renders are stable; comment now tells the truth.
        let bite = std::f32::consts::PI;
        CheeseSpec {
            wheel_radius: 1.0,
            wheel_height: 0.8,
            wedge_bite: bite,
            hole_count: 14,
            hole_radius: (0.06, 0.16),
            hole_seed: bite,
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic scatter
// ---------------------------------------------------------------------------

// Roadmap: an integer bit-avalanche (two multiply/xor-shift rounds) mapped to
// [0,1). It is a hash, not an RNG — referentially transparent, so a given input
// always yields the same output. This is what lets "nonrandom" placement still
// look scattered.
fn swiss_hash(mut h: u32) -> f32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    (h as f32) / (u32::MAX as f32)
}

// Roadmap: derive one bubble (center + radius) purely from (seed, index).
// Placement hugs one of the two wedge cut faces so the sphere is bisected by the
// flat cut and reads as a half-circle in cross-section — the iconic look. Those
// faces sit at +/- wedge_bite/2; for a small bite they border the gap, and for a
// reflex bite they are the flat faces of the surviving wedge — either way the
// bubble lands on a cut surface, which is exactly what we want. The seed's raw
// bits are folded with a per-index Weyl constant (0x9e3779b9) so consecutive
// holes decorrelate. Radial position and height are clamped inward by the chosen
// radius so a bubble never pokes through the rind unexpectedly.
fn bubble(spec: &CheeseSpec, i: usize) -> (Vec3, f32) {
    let base = spec.hole_seed.to_bits() ^ (i as u32).wrapping_mul(0x9e3779b9);
    let which_face = swiss_hash(base ^ 0xA1);
    let t_radial = swiss_hash(base ^ 0xB2);
    let t_height = swiss_hash(base ^ 0xC3);
    let t_size = swiss_hash(base ^ 0xD4);

    let (r_min, r_max) = spec.hole_radius;
    let r = r_min + (r_max - r_min) * t_size;

    // Cut faces sit at +/- wedge_bite/2 around +X (the slice opens toward +X).
    let half = spec.wedge_bite * 0.5;
    let face_angle = if which_face < 0.5 { half } else { -half };

    let rho = (r * 1.2) + (spec.wheel_radius - r * 2.4).max(0.0) * t_radial;
    let half_h = spec.wheel_height * 0.5;
    let y = -half_h + r + (spec.wheel_height - 2.0 * r).max(0.0) * t_height;

    let center = Vec3::new(rho * face_angle.cos(), y, rho * face_angle.sin());
    (center, r)
}

// ---------------------------------------------------------------------------
// The field — "how deep inside the cheese am I?" (negative inside, positive out)
// ---------------------------------------------------------------------------

// Roadmap: exact SDF of a capped cylinder aligned to Y. Work in the two natural
// coordinates: radial distance in XZ and axial distance along Y. The interior
// term (min(max(...),0)) handles points inside; the exterior term is the length
// of the positive overflow in each axis (the standard rounded-box trick applied
// to a cylinder).
fn sd_capped_cylinder(p: Vec3, half_h: f32, r: f32) -> f32 {
    let radial = (p.x * p.x + p.z * p.z).sqrt() - r;
    let axial = p.y.abs() - half_h;
    radial.max(axial).min(0.0)
        + (radial.max(0.0).powi(2) + axial.max(0.0).powi(2)).sqrt()
}

// Roadmap: cheese = cylinder, minus a wedge, minus N bubbles. CSG subtraction of
// B from A is max(A, -B). The wedge bite is symmetric about the z = 0 plane, so
// we fold z by hand (|z|) and take the signed distance to a single rotating cut
// face whose normal is (-sin h, cos h) for half-bite h. The fold makes the two
// physical faces share one expression, and — crucially — it does NOT clamp the
// cos term, so the slice keeps opening correctly as h passes PI/2 (bite past PI).
// The previous max-of-two-planes form was algebraically this same fold but with
// a stray abs() on the cos term, which silently capped the bite at a half-disk
// and turned any reflex request into its supplementary slice. sd_bite is negative
// inside the removed wedge; subtracting it (max against its negation) carves it.
// The result is an SDF near the surface (slightly conservative in concavities —
// fine for sign + gradient meshing).
pub fn cheese_field(spec: &CheeseSpec, p: Vec3) -> f32 {
    let mut d = sd_capped_cylinder(p, spec.wheel_height * 0.5, spec.wheel_radius);

    // Skip the cut entirely for a zero bite: the fold formula below degenerates
    // to sd_bite = |z|, whose negation would carve a zero-width slit through the
    // z = 0 plane. Below this epsilon "round" means round.
    if spec.wedge_bite > 1e-4 {
        let half = spec.wedge_bite * 0.5;
        let folded = Vec2::new(p.x, p.z.abs());
        let cut_face = Vec2::new(-half.sin(), half.cos());
        let sd_bite = folded.dot(cut_face);
        d = d.max(-sd_bite);
    }

    for i in 0..spec.hole_count {
        let (c, r) = bubble(spec, i);
        let sd_sphere = (p - c).length() - r;
        d = d.max(-sd_sphere);
    }
    d
}

// ---------------------------------------------------------------------------
// Polygonizer — generic over ANY field, not just cheese
// ---------------------------------------------------------------------------

const CORNER: [(u32, u32, u32); 8] = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
];
// 12 cube edges as (corner_a, corner_b), grouped x / y / z.
const EDGES: [(usize, usize); 12] = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

// Roadmap: gradient of the field by central differences, normalized. For an SDF
// the gradient points "uphill" = away from the surface = the outward normal, so
// we get smooth shading for free without ever differencing the mesh itself.
fn field_gradient<F: Fn(Vec3) -> f32>(field: &F, p: Vec3) -> Vec3 {
    let e = 0.0015;
    let g = Vec3::new(
        field(p + Vec3::X * e) - field(p - Vec3::X * e),
        field(p + Vec3::Y * e) - field(p - Vec3::Y * e),
        field(p + Vec3::Z * e) - field(p - Vec3::Z * e),
    );
    g.normalize_or_zero()
}

/// Naive surface nets. Roadmap:
///  1. Sample the field at every grid corner ((res+1)^3 values).
///  2. For each cell that the surface passes through (mixed corner signs), place
///     one vertex at the average of its edges' zero-crossings (linear interp).
///  3. For each interior grid edge whose sign flips, the four cells sharing that
///     edge each own a vertex; stitch them into a quad (two triangles). Winding
///     is chosen from the sign direction so faces point outward.
/// Returns (positions, normals, indices) ready to drop into a Mesh.
pub fn surface_nets<F: Fn(Vec3) -> f32>(
    field: F,
    res: usize,
    lo: Vec3,
    hi: Vec3,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>) {
    let d = res + 1;
    let step = (hi - lo) / res as f32;
    let corner_idx = |i: usize, j: usize, k: usize| (i * d + j) * d + k;
    let corner_pos = |i: usize, j: usize, k: usize| {
        lo + Vec3::new(step.x * i as f32, step.y * j as f32, step.z * k as f32)
    };

    let mut vals = vec![0f32; d * d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                vals[corner_idx(i, j, k)] = field(corner_pos(i, j, k));
            }
        }
    }

    let cell = |i: usize, j: usize, k: usize| (i * res + j) * res + k;
    let mut cell_vert = vec![u32::MAX; res * res * res];
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();

    for i in 0..res {
        for j in 0..res {
            for k in 0..res {
                let mut cv = [0f32; 8];
                let mut inside_mask = 0u8;
                for (c, (dx, dy, dz)) in CORNER.iter().enumerate() {
                    let v = vals[corner_idx(i + *dx as usize, j + *dy as usize, k + *dz as usize)];
                    cv[c] = v;
                    if v < 0.0 {
                        inside_mask |= 1 << c;
                    }
                }
                if inside_mask == 0 || inside_mask == 0xFF {
                    continue; // wholly inside or wholly outside -> no surface here
                }

                let mut acc = Vec3::ZERO;
                let mut crossings = 0.0f32;
                for &(a, b) in EDGES.iter() {
                    let (va, vb) = (cv[a], cv[b]);
                    if (va < 0.0) != (vb < 0.0) {
                        let (ax, ay, az) = CORNER[a];
                        let (bx, by, bz) = CORNER[b];
                        let pa = corner_pos(i + ax as usize, j + ay as usize, k + az as usize);
                        let pb = corner_pos(i + bx as usize, j + by as usize, k + bz as usize);
                        let t = va / (va - vb);
                        acc += pa + (pb - pa) * t;
                        crossings += 1.0;
                    }
                }
                let vpos = acc / crossings;
                cell_vert[cell(i, j, k)] = positions.len() as u32;
                positions.push(vpos);
                normals.push(field_gradient(&field, vpos));
            }
        }
    }

    let mut indices: Vec<u32> = Vec::new();
    let mut quad = |a: u32, b: u32, c: u32, e: u32, flip: bool, out: &mut Vec<u32>| {
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || e == u32::MAX {
            return; // a boundary cell never got a vertex; skip the quad
        }
        // CONFESSION: this winding is my best reasoning, not eyeballed in a
        // viewport. If the wheel renders inside-out, swap the two branches
        // below (or set StandardMaterial::cull_mode = None as a quick check).
        if flip {
            out.extend_from_slice(&[a, b, c, a, c, e]);
        } else {
            out.extend_from_slice(&[a, c, b, a, e, c]);
        }
    };

    // x-edges: the 4 sharing cells are offset in j,k
    for i in 0..res {
        for j in 1..res {
            for k in 1..res {
                let s0 = vals[corner_idx(i, j, k)] < 0.0;
                if s0 == (vals[corner_idx(i + 1, j, k)] < 0.0) {
                    continue;
                }
                quad(
                    cell_vert[cell(i, j - 1, k - 1)],
                    cell_vert[cell(i, j, k - 1)],
                    cell_vert[cell(i, j, k)],
                    cell_vert[cell(i, j - 1, k)],
                    s0,
                    &mut indices,
                );
            }
        }
    }
    // y-edges: offset in i,k (note the !s0 — y winds opposite to x and z)
    for i in 1..res {
        for j in 0..res {
            for k in 1..res {
                let s0 = vals[corner_idx(i, j, k)] < 0.0;
                if s0 == (vals[corner_idx(i, j + 1, k)] < 0.0) {
                    continue;
                }
                quad(
                    cell_vert[cell(i - 1, j, k - 1)],
                    cell_vert[cell(i, j, k - 1)],
                    cell_vert[cell(i, j, k)],
                    cell_vert[cell(i - 1, j, k)],
                    !s0,
                    &mut indices,
                );
            }
        }
    }
    // z-edges: offset in i,j
    for i in 1..res {
        for j in 1..res {
            for k in 0..res {
                let s0 = vals[corner_idx(i, j, k)] < 0.0;
                if s0 == (vals[corner_idx(i, j, k + 1)] < 0.0) {
                    continue;
                }
                quad(
                    cell_vert[cell(i - 1, j - 1, k)],
                    cell_vert[cell(i, j - 1, k)],
                    cell_vert[cell(i, j, k)],
                    cell_vert[cell(i - 1, j, k)],
                    s0,
                    &mut indices,
                );
            }
        }
    }

    (positions, normals, indices)
}

// ---------------------------------------------------------------------------
// Bevy glue
// ---------------------------------------------------------------------------

// Roadmap: pick a bounding box that contains the wheel plus a bubble's worth of
// margin, polygonize, then pack the buffers into a Bevy Mesh. UV0 is filled with
// zeros so any StandardMaterial that expects the attribute stays happy; we don't
// texture-map the cheese (yet).
pub fn build_cheese_mesh(spec: &CheeseSpec, resolution: usize) -> Mesh {
    let margin = spec.hole_radius.1 + 0.1;
    let lo = Vec3::new(
        -spec.wheel_radius - margin,
        -spec.wheel_height * 0.5 - margin,
        -spec.wheel_radius - margin,
    );
    let hi = -lo;

    let spec_copy = *spec;
    let (positions, normals, indices) =
        surface_nets(move |p| cheese_field(&spec_copy, p), resolution, lo, hi);

    let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0]; positions.len()];
    let positions: Vec<[f32; 3]> = positions.iter().map(|v| [v.x, v.y, v.z]).collect();
    let normals: Vec<[f32; 3]> = normals.iter().map(|v| [v.x, v.y, v.z]).collect();

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
}

// ---------------------------------------------------------------------------
// The cheese lattice — a 5x5x5 cube of wheels, bite ramping from round to sliver
// ---------------------------------------------------------------------------

/// Cube edge length in wheels. 5 -> 125 cells, one per the user's grand tour.
const GRID: usize = 5;
/// World-space gap between wheel centers. Wheels are radius 1, so ~2.8 leaves a
/// comfortable moat even for the fat-rind near-round ones.
const SPACING: f32 = 2.8;
/// Meshing resolution PER WHEEL. The lone-wheel demo used 96; here we pay it 125
/// times over on a single startup thread, so this is deliberately humbler.
/// CAVEAT for the holiness experiment: the thinnest wheels keep an arc only
/// ~0.025 units wide at the rim, far below this grid's step (~0.05), so the last
/// several cells are UNDER-RESOLVED — they crumble from sampling, not from holes.
/// Distinguishing "holey" from "below the sampling floor" needs res > ~200, which
/// 125-fold on the CPU is a non-starter. Crank this (and expect to wait) or pull
/// the thin endpoint back in `bite_for_cell` if you want a clean verdict.
const GRID_RES: usize = 48;

/// The thinnest few wheels keep so little cheese that the full bubble count
/// crowds into the sliver and reads as more hole than cheese. These two knobs
/// thin the count for just the tail of the ramp: the last THIN_TAIL cells get
/// their hole_count integer-divided by THIN_TAIL_DIVISOR. NOTE this is a hard
/// step, not a ramp, so expect a small visible pop in density at the boundary
/// (cell 117 keeps the full count, 118 onward is halved). Fine for inspection;
/// see the loop comment for the smooth-taper alternative if the pop bothers you.
const THIN_TAIL: usize = 7;
const THIN_TAIL_DIVISOR: usize = 2;

// Roadmap: map a flat cell index n in 0..125 to its removed-wedge angle. A linear
// ramp from 0 (a whole round) at n = 0 up to TAU - PI/125 at n = 124, the latter
// leaving a sliver of just PI/125 radians of cheese. n is laid out so x is the
// slow axis: corner (0,0,0) is the pristine round, opposite corner (4,4,4) = n=124
// is the thinnest sliver, matching "round at one corner, thin slice at the last".
fn bite_for_cell(n: usize) -> f32 {
    let last = (GRID * GRID * GRID - 1) as f32; // 124
    let thinnest_bite = std::f32::consts::TAU - std::f32::consts::PI / 125.0;
    (n as f32 / last) * thinnest_bite
}

/// Marker for every wheel in the lattice, tagged with its cell index so later
/// systems can tell the pristine rounds from the holy slivers.
#[derive(Component)]
struct CheeseWheel {
    #[allow(dead_code)]
    cell: usize,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (lazy_susan, paparazzi))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // One shared material: the meshes are all unique geometry (each bite differs,
    // so no GPU instancing is possible), but they can at least agree on a colour.
    let cheddar = materials.add(StandardMaterial {
        base_color: Color::srgb(0.94, 0.82, 0.35), // a confident cheddar-yellow
        perceptual_roughness: 0.75,
        ..default()
    });

    // Roadmap: three phases.
    //  1. mise_en_place — walk the GRID^3 cube on this thread and lay out every
    //     wheel's recipe (spec + world position). Pure data, cheap; n drives the
    //     bite ramp, (x,y,z) the placement. Hole RECIPE is fixed (constant seed)
    //     so the only thing changing down the diagonal is the wedge angle — the
    //     cleanest read on whether thinning alone makes a wheel holey. (Hole
    //     *angular* position still tracks the bite, since bubbles ride the
    //     +/- bite/2 faces; physical, not a confound.)
    //  2. Mesh all of them in PARALLEL. build_cheese_mesh is the expensive part
    //     (surface nets, single-threaded per wheel), and the wheels are utterly
    //     independent, so we fan the work across the machine's cores via scoped
    //     threads. Mesh is Send, so finished geometry crosses back cleanly, and
    //     the borrow of `mise_en_place` is sound because scope joins all threads
    //     before returning. Chunking by core count (not one-thread-per-wheel)
    //     keeps us from spawning hundreds of threads for a big GRID.
    //  3. Hand the finished meshes to Assets and spawn the entities — back on the
    //     main thread, because Commands and Assets are emphatically not Send.
    let centering = (GRID as f32 - 1.0) * 0.5;
    let total = GRID * GRID * GRID;

    let mut mise_en_place: Vec<(usize, CheeseSpec, Vec3)> = Vec::with_capacity(total);
    for x in 0..GRID {
        for y in 0..GRID {
            for z in 0..GRID {
                let n = (x * GRID + y) * GRID + z;
                let mut spec = CheeseSpec {
                    wedge_bite: bite_for_cell(n),
                    hole_seed: 0.6180339887, // fixed recipe; golden-ish, arbitrary
                    ..default()
                };
                // Thin the bubble budget for the last few slivers. Because
                // bubble(spec, i) is keyed on its index, a smaller count just
                // keeps the same low-indexed SUBSET the fuller wheels show — no
                // reshuffle, the surviving holes sit exactly where they did. NOTE
                // this is a fixed COUNT, so on big grids it covers a shrinking
                // fraction of the truly-thin tail; the angle-based taper (scale
                // hole_count by kept/TAU) is the grid-independent fix.
                if n >= total - THIN_TAIL {
                    spec.hole_count /= THIN_TAIL_DIVISOR;
                }
                let pos = Vec3::new(
                    (x as f32 - centering) * SPACING,
                    (y as f32 - centering) * SPACING,
                    (z as f32 - centering) * SPACING,
                );
                mise_en_place.push((n, spec, pos));
            }
        }
    }

    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk_size = mise_en_place.len().div_ceil(workers).max(1);
    let plated: Vec<(usize, Mesh, Vec3)> = std::thread::scope(|s| {
        let handles: Vec<_> = mise_en_place
            .chunks(chunk_size)
            .map(|batch| {
                s.spawn(move || {
                    batch
                        .iter()
                        .map(|(n, spec, pos)| (*n, build_cheese_mesh(spec, GRID_RES), *pos))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });

    for (n, mesh, pos) in plated {
        commands.spawn((
            CheeseWheel { cell: n },
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(cheddar.clone()),
            Transform::from_translation(pos),
        ));
    }

    // Key light: the bright white directional from upper-front-right. Only its
    // direction matters (directional lights ignore position), and it carries the
    // shadows.
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Fill light: a dim amber wash from the exact opposite side (rear + bottom),
    // so the faces the key leaves in shadow don't crush to black. Directional, so
    // the source itself is invisible — no bulb, no gizmo, just a warm direction.
    // Shadows stay off: a second shadow-caster would duel the key's shadows and
    // cost double for nothing. Illuminance kept well under the key so it reads as
    // a gentle under-glow, not a competing key.
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.57, 0.16), // warm amber
            illuminance: 2000.0,                 // dim; the white key uses the default ~10k lux
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-8.0, -14.0, -10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // The orbit system (`paparazzi`) overwrites this transform every frame, so
    // we only need a sane frame-zero pose: the orbit's start point (theta = 0),
    // which is straight out along +X at the equator. See ORBIT_* for the path.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(ORBIT_A, 0.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

// Gently spin every wheel about Y so the carved faces and cross-sectioned holes
// turn into view. (Dropped the x/z wobble from the single-wheel demo — 125 of
// them tumbling on three axes is a recipe for motion sickness, not inspection.)
fn lazy_susan(time: Res<Time>, mut wheels: Query<&mut Transform, With<CheeseWheel>>) {
    for mut t in &mut wheels {
        t.rotate_y(0.4 * time.delta_secs());
    }
}

// Camera orbit parameters. A gently inclined ("semi-polar") elliptical path
// around the cube's center. A and B are the in-plane semi-axes — deliberately
// unequal so the ring is a slight ellipse, not a circle. INCLINATION tilts that
// ring up off the horizontal toward the poles: 0 is a flat equatorial loop,
// PI/2 would be a true polar pass straight over the top; ~58 deg sits between,
// hence "semi". SPEED is radians/sec, so a lap takes TAU / SPEED ≈ 52 s.
// Both semi-axes sit well outside the cube's far corner (~11.4 units away), so
// the camera never noses inside the lattice. INCLINATION < PI/2 also guarantees
// the camera is never directly above/below the origin, so the Vec3::Y up-vector
// in looking_at never goes degenerate — no gimbal roll at the crests.
const ORBIT_A: f32 = 22.0;
const ORBIT_B: f32 = 19.0; // < A => slightly elliptic
const ORBIT_INCLINATION: f32 = 1.02; // radians, ~58 degrees
const ORBIT_SPEED: f32 = 0.12; // rad/s

// Roadmap: trace an ellipse in its own plane (A·cosθ, 0, B·sinθ), then tilt that
// plane about the X axis by the inclination to lift it toward polar. Rotation
// preserves length, so the orbit radius still ranges only between B and A — the
// tilt redistributes that distance into height. theta is taken straight from
// elapsed time (stateless, perfectly smooth, no accumulator to drift). Each
// frame we rebuild the transform from scratch and re-aim it at the origin.
fn paparazzi(time: Res<Time>, mut cam: Query<&mut Transform, With<Camera3d>>) {
    let theta = ORBIT_SPEED * time.elapsed_secs();
    let (s, c) = theta.sin_cos();
    let (si, ci) = ORBIT_INCLINATION.sin_cos();

    let flat_x = ORBIT_A * c;
    let flat_z = ORBIT_B * s;
    // Tilt the flat ellipse about X: its z-extent splits into height and depth.
    let pos = Vec3::new(flat_x, -flat_z * si, flat_z * ci);

    for mut t in &mut cam {
        *t = Transform::from_translation(pos).looking_at(Vec3::ZERO, Vec3::Y);
    }
}
