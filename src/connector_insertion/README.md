# connector_insertion

Replaces the tiny point-splat connector blobs of the PointWire renderer with real
3D connector models (GLB and similar). The swap happens in 3D: after colours and
textures are applied to the harness point cloud, before rasterisation. The
rasteriser, the lighting bake, the labels and the depth image then treat the new
points like any other point.

The stage is off by default. With it off, every rendered byte is unchanged.

## Quick start

```bash
cd /workspace/kiat_crefle

# 1. Convert the models once (re-run whenever you add or edit one)
PYTHONPATH=src ./env/bin/python -m connector_insertion.build_library

# 2. Render with the stage on (README recipe plus three variables)
KIAT_OUTPUT_ROOT=data/render_3way_conninsert KIAT_DATASET_MODE=phase13 \
KIAT_BG_DIR=data/textures/backgrounds_p4orig11 KIAT_DLO_UNICOLOR=1 \
KIAT_P13_LIGHTING=1 KIAT_CONNINS=1 KIAT_CONNINS_SCALE_MULT=2.0 \
./env/bin/python src/render_full_dataset.py --workers 8
```

`KIAT_CONNECTOR_SCALE` only changes how original class 3 points splat. Inserted
points ignore it unless `KIAT_CONNINS_USE_CONNECTOR_DISC=1`.

## Architecture

```
GLB files ──build_library──> data/connector_models_3d/_library/<name>.npz   (once)

per rendered source (convert_one_video, phase9/11/12/13 only):
  pcl, seg, point_rgb ──> plan_connector_insertion(...) ──> ConnectorInsertion
        slots.detect_slots     cluster seg==3, axis, length, width, wire side
        placement.place_asset  scale, rotate, anchor at the wire end
        colour.tint_and_jitter native colours or wire tint
  convert_to_video_dataset:  harness[~drop_mask] + extras appended to the combined cloud
  apply_scene_lighting       lights the extras together with everything else
  rasterize_view             label INSERTED_CONNECTOR_LABEL -> PNG label 4, z-buffered
```

| Module | Role |
|---|---|
| `constants.py` | `INSERTED_CONNECTOR_LABEL` (103), pixel scale, RNG offset. The only place the label is defined. |
| `config.py` | `ConnectorInsertionConfig`, a frozen validated dataclass with every knob. `from_env`, `from_dict`, `from_json`. |
| `library.py` | Reads the `.npz` library (numpy only). Include/exclude globs and weights. |
| `assets.py` | Mesh loading, textured surface sampling, PCA canonicalisation, outer-shell filter (needs trimesh). |
| `build_library.py` | CLI that builds the library and maintains `manifest.json`. |
| `slots.py` | Finds connector slots in a harness cloud. |
| `placement.py` | Pure geometry: size fit and rigid transform of an asset into a slot. |
| `colour.py` | Wire tint and colour jitter. |
| `insert.py` | `plan_connector_insertion`, the one entry point. Never mutates its inputs. |
| `tests/` | `unittest` suite. |

`plan_connector_insertion(pcl, seg, point_rgb, cfg, rng, library)` returns an
immutable `ConnectorInsertion` with `drop_mask` (bool over the harness points),
`extra_pts`, `extra_rgb`, `extra_labels` and a `report` dict. When the stage is
disabled, the library is empty or no slot exists, it returns an empty result.

The persisted per-point `pointclouds/*.npy` and `labels/*.npy` stay (4096,) and
unchanged. When the stage is on, each source also gets
`<split>/<set>/connector_insertion/<frame>.json` with the planner report, and
`metadata.json` gains a `connector_insertion` block with the full config.

## Canonical asset frame

Every library file stores one asset as a coloured surface point cloud with
centre at the origin, long axis on +X, long extent exactly 1.0, cable end at -X
and mating end at +X. Y is the wider cross-section direction. Colours are uint8
BGR. Placement maps canonical -X onto the direction from the connector toward
its wire, so the cable end always faces the wire and the cable side face sits at
the wire end of the slot.

## Knobs

Every field of `ConnectorInsertionConfig` is also an env var named
`KIAT_CONNINS_<FIELD>`. The master switch is `KIAT_CONNINS`. Lists are comma
separated, weights are `name=w,name=w`.

| Field | Default | Meaning |
|---|---|---|
| `enabled` (`KIAT_CONNINS`) | off | Master switch. |
| `library_dir` | `data/connector_models_3d/_library` | Library folder. |
| `include`, `exclude` | all, none | fnmatch globs on asset names. |
| `weights` | 1 each | Per-asset sampling weights. |
| `replace_prob` | 1.0 | Chance that each slot is replaced. |
| `max_per_frame` | 24 | Cap on replaced slots per source, largest clusters first. |
| `fit_mode` | `length` | Fit the asset length to the slot length, its width to the slot width, or `min`, `max`, `geomean` of the two scales. |
| `scale_mult` | 1.0 | Global size multiplier (the bulky connector cue). |
| `scale_jitter` | 0.0 | Uniform +- fraction on the size. |
| `min_length_px`, `max_length_px` | 12, 48 | Clamp on the final connector length in pixel equivalents. |
| `px_per_unit` | 218.18 | World units to pixels (renderer SCALE). |
| `roll_mode` | `random` | Roll about the connector axis: `random`, `fixed` (uses `roll_deg`), `none`. |
| `roll_deg` | 0 | Roll for `fixed`. 0 puts the asset's wide side along the slot's wide direction. |
| `tilt_jitter_deg` | 0 | Max random tilt of the axis about the cable face. |
| `flip_prob` | 0 | Mirror probability. |
| `wire_overlap_frac` | 0 | Fraction of the length pushed into the wire. |
| `wire_tint` | 0 | 0 keeps native colours, 1 recolours to the frame's mean wire colour. |
| `asset_color_jitter` | 0 | +- fraction of brightness and per-channel gain. |
| `spacing_px` | 0.5 | Target splat spacing at the final size. |
| `min_points`, `max_points` | 300, 8000 | Point budget per connector. |
| `cluster_eps` | 0.05 | DBSCAN eps (world units) for slot detection. |
| `min_cluster_points` | 8 | Smaller clusters are ignored. |
| `split_large`, `split_length_px` | off, 40 | KMeans split of merged clusters longer than the threshold. |
| `use_connector_disc` (`KIAT_CONNINS_USE_CONNECTOR_DISC`) | off | Splat inserted points with the `KIAT_CONNECTOR_SCALE` disc. This one is read by `pcl_to_rgbd.py`. |
| `seed_salt` | 0 | Added to the planner seed. |

Randomness comes from `RandomState(set_id*1000 + frame_id + 733 + seed_salt)`,
which no other renderer stream shares. Each slot draws a fixed number of values,
so changing a knob does not reshuffle the choices of other slots.

## Colour and the colour cue

With `KIAT_DLO_UNICOLOR=1` the whole harness, connectors included, carries the
wire texture. Native-coloured inserted connectors bring a colour difference
between connector and wire back. Real connectors do differ from their cable, so
this is a design choice. `wire_tint` sets how much of that difference remains:
0 keeps it, 1 removes it while keeping the asset's light and dark structure.

## Adding models

1. Copy the file (`.glb`, `.gltf`, `.obj`, `.stl`, `.ply`) into `data/connector_models_3d/`.
2. Run the build command above. A stub entry appears in `manifest.json`.
3. Fill in `source_url`, `creator`, `license`. Set `license_verified` to true only after checking the source page.
4. Look at `results/connector_insertion/make_library_sheet.py` output (or any scatter plot of the `.npz`). If the cable end is wrong, set `cable_end` in the manifest to the direction in the model's own coordinates that points toward the cable (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`) and rebuild with `--force`.

The builder samples 400k area-uniform points, reads `baseColorTexture` through
UV barycentric interpolation (textures above 1024 px are downscaled), falls back
to the material colour, and drops samples that no direction in 32 views can see
(`--no-shell` keeps them). Cable-end auto detection picks the narrower end of the
long axis. That heuristic is right for the three current assets and can fail on
models without a strain relief, hence the override.

## Zoom views (opt-in close-ups)

The standard renders are orthographic at 218 px/unit, so a connector is 12 to 48 px long. Zoom views add N
extra 640x480 close-ups per source, centred on a connector slot, at a larger pixel scale. Default OFF
(`KIAT_CONNINS_ZOOM_VIEWS=0`): every existing render is byte-identical. Needs `KIAT_CONNINS=1`, a static
phase12/phase13 render and at least one slot (sources without a slot get no zoom files, or a random wire
location with `zoom_fallback=wire`).

```bash
KIAT_CONNINS=1 KIAT_CONNINS_SCALE_MULT=2.0 KIAT_CONNINS_ZOOM_VIEWS=3 ... python src/render_full_dataset.py
PYTHONPATH=src ./env/bin/python -m connector_insertion.zoom_preview --set 0 --frame 5 --zoom 4 --asset usb-a_tip
```

How a zoom view is built (`zoom.py`, hook after the six standard views in `convert_one_video`):

- Camera: a canonical view rotation (or a perturbed one), centred on the slot with a bias toward the wire-side
  end plus jitter, zoom chosen so the connector's projected length is a random fraction of the frame height,
  clamped to `[zoom_min, zoom_max]`. Views that foreshorten the connector are skipped; up to `zoom_max_attempts`
  candidates are tried until at least `zoom_min_visible_frac` of the connector is unoccluded.
- `rasterize_view(..., zoom, centre, point_radius)`: defaults are an exact no-op. Depth encoding is unchanged.
  Zoomed splatting is vectorised, window-culled and uses a stable depth sort. Splat radius is `ceil(zoom)` for
  scene points, `zoom_connector_radius` (1) for connector points, `zoom_wire_splat_radius` (2) for tube points,
  1 for class-4 noise specks.
- Connectors are re-sampled from the library at `zoom_connector_spacing_px` on screen (same pose, size and colours;
  the sparse points are a prefix of the dense ones). The library has 150k points per asset, never upsampled.
- Wire: with `zoom_wire_densify` the harness wire points are replaced by a smooth hollow tube along the smoothed
  skeleton (radius per node from the point spread, caps at free ends, camera-facing half only, colours
  inverse-distance weighted from the lit harness points). The tube step is `zoom_wire_step_px` (default 1.5 px, not
  0.5: a hollow tube at 0.5 px would need about 10x the points).
- Floor in front: from the canonical `bottom` camera the 3D floor points sit between camera and harness, and their enlarged splats would be drawn over the wire and connector. `bottom` is excluded by default. For every candidate view the renderer also counts background points in the window that are nearer than the nearest harness point; candidates with more than 5 are tried last, and any such points are culled from the zoom cloud (a hole in the floor is invisible, the floor is the nearest layer there). The count and the culled number are in the per-view JSON (`occluding_bg_points`, `culled_background_points`).
- Lighting: not re-run. Scene points keep their baked colours; re-sampled connector points go through the same
  `apply_scene_lighting` call (it only depends on world position).
- Backdrop: `zoom_backdrop_mode=crop` crops the camera window of the 2D photo and resizes it bicubically, the image a
  zoom-times larger render would show. `static` keeps the unzoomed photo.
- Outputs: `{src:04d}_{anim:02d}_zoom{j}.png` in `rgb/`, `depth/`, `label/`; the per-source JSON gains a
  `zoom_views` list (view, rotation, centre, zoom, target and planned fraction, visibility, attempts, seconds);
  `metadata.json` gains `zoom_views.names`. Zoom names are NOT added to `views`.
- Staging: `prepare_dformer_data.py --include-zoom` stages them (default off, files missing for slotless sources
  are skipped silently). `reencode_labels_3way.py` needs no change. A zoom view belongs to its source, so it
  follows the source's set-level split and the `--src-frame-step` / `--src-stride` subsampling (no leakage).
- Randomness: `RandomState(set_id*1000 + frame_id + 1531 + seed_salt)`, created only when `zoom_views > 0`; each
  view draws a fixed number of values.

| Field | Default | Meaning |
|---|---|---|
| `zoom_views` | 0 | Close-ups per source, 0 = OFF. |
| `zoom_target_frac_min/max` | 0.10 / 0.45 | Connector projected length over frame height, uniform draw. |
| `zoom_min/max` | 1.5 / 8.0 | Clamp on the zoom factor. |
| `zoom_centre_jitter` | 0.08 | Centre jitter, fraction of frame height. |
| `zoom_centre_bias` | 0.3 | Centre along the connector, 0 cable end, 1 tip. |
| `zoom_rotation_mode`, `zoom_perturb_deg` | canonical, 15 | `canonical` or `perturb` (canonical plus random rotation). |
| `zoom_min_projected_frac` | 0.6 | Skip views foreshortening the connector below this share of its best view. |
| `zoom_max_attempts`, `zoom_min_visible_frac`, `zoom_min_connector_px` | 3, 0.6, 100 | Occlusion retry rule. |
| `zoom_wire_densify` | on | Smooth tube instead of chunky harness points. |
| `zoom_wire_step_px`, `zoom_wire_splat_radius` | 1.5, 2 | Tube sampling and splat. |
| `zoom_bg_radius_mult` | 1.0 | Scene splat radius `ceil(zoom x mult)`. |
| `zoom_backdrop_mode` | crop | `crop` or `static`. |
| `zoom_connector_spacing_px`, `zoom_max_points`, `zoom_connector_radius` | 0.7, 150000, 1 | Connector sampling. |
| `zoom_exclude_views` | bottom | Canonical views never used as the zoom camera base (comma separated env list). |
| `zoom_fallback` | skip | `skip` or `wire` for sources without a slot. |

Env var for each is `KIAT_CONNINS_<FIELD>`. Limits: the scene (floor, clutter, speckle floor points) is point splats
and looks mosaic or dotted at 5x and above; the 2D backdrop is blurry at high zoom; `KIAT_CONNECTOR_SCALE` discs and
the soft-edge wire pass (`KIAT_P26_SOFTEDGE`) are not applied in zoom views; the connectors are mostly flat dark
plastic with no per-point shading.

## Tests

```bash
cd /workspace/kiat_crefle
PYTHONPATH=src ./env/bin/python -m unittest discover -s src/connector_insertion/tests -t src -v
```

## Limits

Static renders only (`num_frames == 1`). The v2 animated mode would need the
models to follow the skeleton, and the feature raises there. Wire-free frames
skip the planner. Models carry flat colours or textures without lighting
information beyond what `apply_scene_lighting` bakes, so dark plastic reads as a
dark silhouette at 12 to 25 px.
