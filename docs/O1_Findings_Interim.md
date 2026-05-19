# Infinity Shield

## Objective O1 — Detection Benchmarking and Explainability: Interim Findings

MEng Final Year Project — University of Ghana

*Status: interim draft. Two of three detectors complete; third detector and unified metric reconciliation pending.*

---

## 1. Scope of This Document

This document records interim findings for Objective O1, which benchmarks lightweight object detectors for real-time plastic-projectile detection on edge hardware and applies explainable-AI (XAI) techniques to interpret their behaviour.

Two detectors have been trained and analysed to date: YOLOv8n and MobileNet-SSD (SSDLite320-MobileNetV3-Large). A third detector is planned; its identity is the subject of a pending scope decision with the project supervisor (see Section 6). The metric figures reported here are preliminary and require the reconciliation step described in Section 5 before they can be presented as directly comparable.

## 2. Method Summary

Both detectors were trained on the same plastic-ball dataset (Roboflow export, YOLOv8 label format, single class, 80/10/10 split, approximately 1,200 images after augmentation). YOLOv8n was trained at 640 px input resolution using the Ultralytics framework; MobileNet-SSD was trained at its native 320 px input using torchvision with full fine-tuning over 50 epochs.

### 2.1 Annotation Convention

Only balls in flight were annotated as positive instances. Balls held in a person's hand were intentionally left unannotated and therefore treated as negatives. This reflects the system's purpose: Infinity Shield responds to projectiles in flight, not stationary held objects. Consequently, throughout this document a held ball with no detection box is correct behaviour and is not counted as a missed detection.

## 3. Quantitative Results (Preliminary)

*The following figures are recorded as measured. They are not yet directly comparable because the two detectors' metrics were produced by different measurement procedures — see Section 5.*

| Metric | YOLOv8n | MobileNet-SSD |
|---|---|---|
| Input resolution | 640 × 640 px | 320 × 320 px |
| Test recall | 0.82 | 0.50 |
| Precision | 0.94 | 0.78 (derived: TP/(TP+FP) at 0.40 conf.; to be confirmed by metric reconciliation) |
| mAP@0.5 | 0.91 | Not yet computed |
| False-positive rate | Low (see Section 4) | 0.22 |
| Latency, mean (laptop CPU) | ~144 ms | ~53 ms |

*Note: CPU latency figures are baseline only. Target-hardware (Jetson / Raspberry Pi 5) measurements will replace these once hardware is available.*

## 4. Qualitative Analysis (8-Image Sample)

Eight representative test images were examined in detail, combining each detector's predicted boxes with EigenCAM saliency maps. Each predicted box was manually classified as a true positive (box on a real in-flight ball), false positive (box on a non-ball), or duplicate (a second box on an already-counted ball); missed in-flight balls were also recorded. This sample is qualitative and corroborative; it illustrates the quantitative results of Section 3 rather than replacing them.

### 4.1 Per-Image Classification

| Img | In-flight balls | YOLOv8n | MobileNet-SSD |
|---|---|---|---|
| 1 | 0 | Correct — no detection (true negative) | False positive on a reflective door fitting (conf. 0.44) |
| 2 | 1 | True positive (conf. 0.76) | True positive (conf. 0.82) |
| 3 | 1 | True positive, box fully on ball (conf. 0.41) | True positive, box only partly on ball (conf. 0.25) |
| 4 | 1 | True positive (conf. 0.67) | True positive (conf. 0.97) |
| 5 | 2 | Two true positives, one box on each ball (conf. 0.72, 0.60) | Both boxes on the rear blurred ball: one true positive, one duplicate; the nearer ball was missed (conf. 0.59, 0.34) |
| 6 | 1 | True positive (conf. 0.74) | True positive (conf. 0.95) |
| 7 | 1 | True positive (conf. 0.94) | True positive (conf. 0.46) |
| 8 | 1 | True positive (conf. 0.81) | True positive (conf. 0.98) |

### 4.2 Sample Totals

Across the eight images (eight in-flight balls present):

- YOLOv8n: eight true positives, no false positives, no duplicates, no missed balls, and one correct true negative.
- MobileNet-SSD: six true positives plus one partial-box true positive, one false positive, one duplicate, and one missed ball.

*The sample is small and is presented as qualitative corroboration only; it does not constitute a precision or recall measurement.*

### 4.3 Confidence Versus Detection Difficulty

A notable pattern emerged when detection confidence was compared against how difficult each ball was for a human observer to see (clear and static versus motion-blurred or partially out of frame).

On the single clear, fully visible, static ball in the sample (image 7), YOLOv8n reported 0.94 confidence — its highest — while MobileNet-SSD reported 0.46 — among its lowest. Conversely, MobileNet-SSD assigned 0.95 to 0.98 confidence to motion-blurred and partially out-of-frame balls (images 4, 6, 8), which are the hardest cases in the set.

YOLOv8n's confidence therefore tracks genuine detection difficulty: it is most certain on the easiest target and less certain on harder ones. MobileNet-SSD's confidence is inverted relative to difficulty. This inversion indicates that MobileNet-SSD's confidence is driven by similarity to memorised training patterns rather than by genuine object understanding, which is consistent with the overfitting identified during its training (training loss fell to near zero while validation recall plateaued by epoch 12).

### 4.4 Explainability (EigenCAM)

EigenCAM was selected as the saliency method because it requires no class-specific target and is therefore robust for object-detection architectures. AblationCAM was also evaluated but proved incompatible with YOLOv8's routed network architecture and was not pursued further.

For most images, EigenCAM saliency was diffuse and concentrated on high-contrast scene structure (wall edges, doorframes, the person) rather than on the ball itself. This is an expected property of EigenCAM, which highlights a layer's dominant feature components rather than class-specific evidence. The maps are therefore interpreted as a diagnostic of where each network's strongest activations lie, rather than as direct proof of object localisation. One MobileNet-SSD case (image 8) produced saliency concentrated on the ball; this is reported as a single favourable instance and not generalised.

## 5. Outstanding: Metric Reconciliation

The two detectors' metrics were produced by different procedures. YOLOv8n's recall, precision and mAP came from the Ultralytics validation routine, which sweeps confidence thresholds and computes COCO-style mAP. MobileNet-SSD's recall and false-positive rate came from a fixed-threshold counting procedure at 0.40 confidence. These are not directly comparable, and MobileNet-SSD has no mAP value yet.

Before the final O1 results are presented, all detectors will be re-evaluated on the test set using a single consistent implementation (torchmetrics, COCO-style mean average precision). This is a test-only evaluation and requires no retraining. It is deferred until the third detector is trained so that all three models can be measured together in one pass.

## 6. Outstanding: Third Detector

Objective O1 as originally approved specifies three detectors: YOLOv8n, MobileNet-SSD and NanoDet. During implementation, NanoDet was found to require an older software stack (PyTorch below version 2.0) that conflicts with the environment used for the other two detectors, and its installation process is known to be fragile.

A substitution — replacing NanoDet with a torchvision-based detector such as Faster R-CNN MobileNetV3-FPN — has been identified as a lower-risk alternative that also reuses the existing pipeline and offers a one-stage versus two-stage architectural contrast. This substitution is a change to an approved objective and is pending the project supervisor's approval before any work proceeds. This section will be completed once that decision is made.

**[Placeholder — third detector results, qualitative analysis, and three-way comparison to be added here.]**

## 7. Interim Conclusions

On the evidence gathered so far, YOLOv8n is the stronger and more reliable detector for this task. It achieved higher test recall (0.82 versus 0.50), and on the qualitative sample it placed every box correctly with no false positives or duplicates, including on motion-blurred and partially visible balls. Its detection confidence correctly reflects detection difficulty.

MobileNet-SSD detects plastic balls but is unreliable. It produced a confident false positive on a ball-like distractor, a duplicate detection, and a missed ball (notably the easier of two), and its confidence does not track genuine difficulty. Both its measured metrics and its qualitative behaviour point consistently to overfitting on the small dataset, compounded by its lower 320 px input resolution. This is a legitimate and useful comparative finding for O1: it establishes a clear accuracy gap between a current-generation detector and a lightweight older-style one under identical data conditions.

*These conclusions are interim. They are subject to the metric reconciliation of Section 5 and the addition of the third detector in Section 6.*
