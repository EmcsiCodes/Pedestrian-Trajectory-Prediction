# Project and Design Plan — Phase 1

**Project:** Pedestrian Trajectory Prediction  
**Working thesis title:** Context-Aware Pedestrian Trajectory Prediction from Video  
**Version:** 0.3 · 13 November 2026  
**Status:** Initial design; implementation and experiments pending.

## 1. Objective

Build and train a custom model that predicts several complete future pedestrian trajectories using movement history, body pose, scene geometry, and nearby people's observed movement. Integrate it into a video pipeline and compare its behavior with simple baselines and an established forecasting model.

The central research question is:

> How much do pose, scene geometry, and neighboring pedestrians improve direct prediction of complete future paths over motion history alone, and what accuracy and runtime trade-offs does our design achieve relative to an established predictor?

The intended contribution is our forecasting architecture, training implementation, controlled experiments, and analysis. Existing models already use motion, context, multiple futures, and pose. A claim of novelty must identify a specific difference through related-work review and evidence.

## 2. Agreed direction and open decisions

| Agreed direction | Meaning |
| --- | --- |
| Direct complete-path prediction | Output K complete future position sequences with ranking scores. Endpoints are the last positions in those sequences. |
| Custom forecasting model | Implement and train our own design using standard neural-network building blocks. |
| Four planned input sources | Target motion, target pose, scene geometry, and nearby people's observed motion. All four belong to the intended full model. |
| Existing perception components | Reuse detection, pose estimation, and tracking implementations to obtain observations. |
| Independent comparison | Use constant velocity, a motion-only learned baseline, and an established predictor; Y-Net is the proposed learned reference. |
| Phase 1 is design and feasibility | Establish a data contract and a defensible first architecture before full implementation and training. |

Still to decide: dataset, camera viewpoint, coordinate convention, observation duration, prediction horizon, sampling interval, K, missing-data policy, encoder sizes, fusion mechanism, decoder details, and training objective. GPU, memory, and available development time must also be recorded.

The initial preference is to output every future time step in one forward pass. The direct-path decision does not require a particular layer type; an autoregressive decoder would also predict paths directly, but is not the starting candidate.

## 3. Scope and ownership

The finished application should process recorded video, show detections and track IDs, draw observed motion, display K predicted paths, and later reveal the actual future without changing the saved forecast. It should export predictions, configurations, metrics, and representative failures.

Tracking supplies identity-consistent histories. Developing a new identity-association or re-identification algorithm is outside this project's scope. Other exclusions are robot control, multi-camera identity matching, full 3D reconstruction, and future-video generation. Live-camera support can follow a working recorded-video demonstration.

| Component | Reuse, build, or adapt |
| --- | --- |
| Person detector | Start from pretrained YOLO weights. Plan a bounded fine-tuning comparison once suitable training boxes and a validation split exist. |
| Pose estimator | Integrate a pretrained model with keypoint confidence. A pose model may provide both boxes and keypoints, avoiding a redundant detector pass. |
| Tracker | Reuse an existing implementation; ByteTrack is an initial candidate. |
| Scene representation | Use available aligned maps or derive a documented walkability/obstacle representation. |
| Forecasting network | Build our own encoders, fusion, decoder, losses, training loop, and checkpoints. Forecasting weights will be trained on the selected training split. |
| Reference predictor | Integrate Y-Net as a separate model, checking its preprocessing and training-data provenance. |
| Data and application | Build preprocessing, coordinate transforms, caches, evaluation, visualization, and reproducibility records. |

Detector fine-tuning is a supporting experiment, not a prerequisite for designing the predictor. Pose fine-tuning would require keypoint labels; box annotations alone do not supervise a skeleton model.

## 4. Forecasting contract

For one target pedestrian at prediction time t, use only information available at or before t.

| Field | Proposed meaning |
| --- | --- |
| Target history | T_obs positions, timestamps, validity masks, and optionally derived velocities. |
| Pose history | J body keypoints per observed time step, with confidence and visibility masks. |
| Scene context | An aligned scene image or walkability/obstacle map, with the crop and coordinate transform recorded. |
| Neighbor histories | Up to N nearby people's observed positions or velocities, relative to the target, with masks. |
| Training target | T_pred annotated future positions of the same pedestrian. These are labels, never model inputs. |
| Prediction | K trajectories, each with T_pred future 2D positions, plus K ranking scores. |

The batched output shapes are `trajectories: [B, K, T_pred, 2]` and `scores: [B, K]`, where B is batch size. K and both sequence lengths are configuration values still to be selected.

Positions, neighbors, and scene geometry must share a documented coordinate frame. Choose a position convention such as a box reference point or calibrated ground-plane location based on the dataset. Normalized image coordinates are not metres. If the camera moves, ego-motion compensation and map alignment become explicit requirements.

Represent target motion relative to its last observed position where useful; preserve the inverse transform. Normalize body pose in a separate body-centered frame so image scale does not dominate joint features. Transform movement and maps consistently when augmenting data.

Never feed future neighbors, future poses, true test endpoints, or histories smoothed using later frames into the predictor. Dataset annotation interpolation must be audited for this issue.

## 5. Candidate architecture

These components define a first design to evaluate, not a requirement to implement several competing architectures.

| Module | Starting candidate | Design question to resolve |
| --- | --- | --- |
| Motion encoder | Small GRU over positions/displacements and valid time steps. | Which motion features and history duration carry enough information? |
| Pose encoder | Per-frame joint-feature network followed by a small temporal encoder. | How should confidence, missing joints, and body orientation affect the representation? |
| Scene encoder | Compact CNN producing a spatial feature map. | What crop, map channels, and spatial resolution capture usable geometry? |
| Interaction encoder | Shared temporal encoding of neighbors, followed by target-conditioned attention. | Which neighbors are included, and how are absent neighbors masked? |
| Fusion | Target-conditioned learned fusion of motion, pose, interaction, and spatial scene features. | Can the decoder retain access to where obstacles are located? |
| Direct decoder | K candidate embeddings with a shared decoder, or K output heads, each producing the complete future sequence. | Which implementation learns distinct, useful alternatives without excessive cost? |
| Ranking head | One logit per candidate, optionally normalized with softmax. | Does the preferred candidate predict well, and are scores meaningful? |

Start the motion-and-decoder design with a small GRU and direct coordinate/displacement outputs. Keep spatial scene features available to fusion or the decoder instead of prematurely compressing the entire map into one vector. Attention is one candidate mechanism, not a commitment to a large transformer.

The output can represent offsets from the last observed location or future step displacements accumulated into positions. Phase 1 must choose one and define the conversion used by training, evaluation, and visualization.

Implementing a motion-only version first will help validate this contract. It is an experimental baseline and development step; pose, scene, and interactions remain part of the planned full model.

### Training multiple futures

Usually each example contains one realized future, even when several futures were possible. Applying the same regression target independently to every candidate can make all outputs converge to similar paths.

One starting objective to investigate is:

1. Compute a full-trajectory error for each candidate against the observed future.
2. Regress the best-matching candidate toward that future.
3. Train the ranking head to favor that candidate.
4. Inspect candidate usage, diversity, and plausibility before adding regularization or changing the objective.

This is a candidate winner-takes-all objective with a ranking loss, not a guarantee of diverse predictions. Unused candidates and mode collapse must be checked. Do not reward arbitrary separation that pushes paths into obstacles. Any scene or smoothness penalty must have a separate ablation.

Ranking scores are not automatically calibrated probabilities. Alternative paths cannot all be declared correct or incorrect from one recorded future. The final objective and its justification remain a Phase 1 decision.

## 6. Dataset selection

Choose data that supports the intended model, not just a visually appealing walking video. Audit a small sample before committing to a complete download or preprocessing effort.

Required checks:

- Video frames, reliable timing, pedestrian identities, and annotated positions covering observations and futures.
- People large and visible enough for useful pose features, or aligned pose annotations.
- Scene/map information that can be aligned with the trajectories.
- Simultaneous neighbor observations and meaningful pedestrian interactions.
- Clear handling of occlusion, missing labels, identity changes, and camera movement.
- Enough independent sequences/scenes for training, validation, and testing.
- Practical access, storage, permitted use, and compatibility with the reference model.

[Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) is worth auditing for video, trajectories, and Y-Net compatibility. Its aerial viewpoint makes pose usability a specific question to test. [JRDB-Pose](https://jrdb.erc.monash.edu.au/dataset/pose) is another candidate for pose-rich observations, but its mobile-camera setting adds alignment and ego-motion work. Neither is selected yet.

If a candidate cannot support a cue, record the evidence and reconsider the dataset or document a deliberate scope decision. Do not silently remove pose or social context from the full model.

## 7. Evaluation design

Separate two questions:

1. **Forecasting quality:** compare predictors on the same eligible annotated histories and future labels, with the provenance of pose/map features recorded.
2. **Pipeline quality:** repeat with detector/tracker histories, keeping future annotations for scoring and reporting tracking coverage and missing forecasts.

| Comparison | Purpose |
| --- | --- |
| Constant velocity | Establish a simple extrapolation reference. |
| Custom motion-only predictor | Establish the learned baseline with the same trajectory output format. |
| Full custom model minus pose, minus scene, and minus neighbors | Isolate the contribution of each input source. |
| Full custom model vs Y-Net | Compare with an established forecasting approach. |
| Original vs fine-tuned detector, once available | Measure perception adaptation and its downstream effect separately. |

For Y-Net, document both the input information available to each model and a matched-information comparison where practical. Improvements from extra pose or neighbor information must not be attributed solely to architectural changes. Verify that any pretrained checkpoint has not trained on the intended test data; otherwise retrain on the approved split or restrict the claim.

Planned metrics:

- **ADE/FDE of the top-ranked candidate:** mean position error over the horizon and error at its final step.
- **minADE@K and minFDE@K:** best candidate in hindsight, with the same K and sampling budget across comparable models. These are oracle measures, not the application's actual selection; the two minima may select different candidates.
- **Candidate behavior:** duplicate paths, candidate usage, diversity, and coverage of observed futures.
- **Scene consistency:** obstacle or non-walkable-region intersections where reliable aligned maps exist. Define how segments between predicted points are checked.
- **Efficiency:** model latency, parameter count, and full-pipeline runtime on recorded hardware and batch settings.

Use physical-distance or pedestrian-collision claims only when calibration and evaluation definitions support them. A soft obstacle penalty does not guarantee safe or collision-free behavior.

Split by scene or independent sequence before constructing overlapping windows. Keep all related frames and identities within the chosen partition. Select hyperparameters on validation data and reserve test data for the final evaluation. Record units, horizons, data availability, random seeds, checkpoint provenance, and preprocessing versions for every comparison.

## 8. Phase 1 work and completion criteria

Phase 1 ends with a specific, feasible design and evaluation contract. Full training runs and a finished application belong to later phases.

| Step | Work | Evidence required to finish |
| --- | --- | --- |
| 1. Record constraints | Capture hardware, storage, environment, and available development time. | A short resource note that limits model and dataset choices. |
| 2. Inspect candidate data | Review a few sequences and overlay trajectories, estimated/annotated pose, and scene coordinates. | A dataset choice with examples, known limitations, and a split strategy. |
| 3. Specify one sample | Define every input, output, mask, coordinate transform, and time interval. | One annotated example and a complete tensor/data contract. |
| 4. Resolve the model design | Select the initial encoder, fusion, decoder, ranking, and loss implementations. | An architecture diagram and concise reasons for each choice; no unresolved output semantics. |
| 5. Check comparison feasibility | Inspect and minimally run the reference implementation on suitable sample data where possible; identify preprocessing and checkpoint constraints. | A reproducibility note for Y-Net, or a documented reason to select another established predictor. |
| 6. Freeze the first experiment | Specify the split, baselines, ablations, metrics, units, and compute budget. | A reproducible experiment specification ready for implementation. |

Completion checklist:

- [x] Main task and direct-path output selected.
- [x] Four-source candidate architecture documented.
- [x] Repository overview and Phase 1 plan updated.
- [ ] Dataset selected after sample inspection.
- [ ] Coordinates, timing, masks, and missing-observation policy fixed.
- [ ] T_obs, T_pred, sampling interval, and K fixed for the first experiment.
- [ ] First model and training objective specified.
- [ ] Reference feasibility and training-data provenance checked.
- [ ] Split, metrics, ablations, and hardware recorded.

**Immediate next action:** inspect candidate dataset samples for usable pose and aligned trajectories, then define one training example end to end. This resolves the largest dependency before committing to detailed layers or training schedules.

After Phase 1, development should proceed from a verified data loader and motion-only predictor to the full model, controlled experiments, perception integration, and a replayable demonstration. Detailed milestones will follow the selected data and design.

## 9. Reading tied to decisions

- [Y-Net and PECNet: official code and papers](https://github.com/HarshayuGirase/Human-Path-Prediction) — understand the established reference and its data protocol; its goal-based design does not determine our architecture.
- [Human Scene Transformer](https://human-scene-transformer.github.io/) — related work on using pose information for trajectory forecasting.
- [Ultralytics pose estimation](https://docs.ultralytics.com/tasks/pose/) — understand keypoint outputs, confidence, and training-label requirements.
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) — understand how existing tracking components provide observation histories.

## Decision history

Version 0.3 replaces the earlier identity-association scope and the provisional endpoint-first design with direct prediction of complete paths. It restores all four input sources to the intended full model and keeps Phase 1 focused on design and feasibility. Earlier plans remain available through version history.
