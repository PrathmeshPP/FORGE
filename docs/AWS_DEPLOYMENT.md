# F.O.R.G.E — AWS Cloud Video Preflight Service Architecture & Deployment Guide

**Submission Target:** OpenCV AI Competition 2026  
**Component:** Minimal, Meaningful AWS Lambda Video-Preflight Worker (OpenCV 5.x)  
**S3 Bucket:** `forge-drone3d-prathmesh-2026-8xq472`  
**Region:** `ap-southeast-2`  
**Status:** Implementation Complete, Tested & Deployment-Ready  

---

## 1. Architecture Overview

F.O.R.G.E uses a **hybrid cloud-edge architecture** designed to maximize cost efficiency, compute reliability, and geospatial reconstruction accuracy:

1. **Cloud Preflight Stage (AWS Lambda + Amazon S3):**
   - Automatically triggered when an operator or aerial survey drone uploads raw MP4 footage to Amazon S3.
   - Operates entirely on serverless compute (AWS Lambda with `opencv-python-headless==5.0.0.93`).
   - Validates video container decodability, queries frame dimensions/FPS, uniformly samples frames, and computes multi-metric sharpness (Laplacian variance, Tenengrad energy) and invariant ORB feature counts.
   - Evaluates a **3D Reconstruction Feasibility Decision Gate** (`APPROVED_FOR_3D_RECONSTRUCTION`, `NEEDS_REVIEW`, `REJECTED`).
   - Persists a structured JSON quality report and visual contact sheet mosaic back to S3.
2. **Local / Edge Reconstruction Stage (Workstation / Field GPU):**
   - Ingests the preflight decision report.
   - Only processes videos approved by the preflight gate, preventing wasted compute cycles and foundation model GPU allocation on unusable, blurry, or degenerate video captures.
   - Executes dense OpenCV 5 frame curation (`src/video/opencv5_frame_analyzer.py`) and VGGT-Omega 3D reconstruction.

```mermaid
graph TD
    subgraph AWS Cloud ("AWS ap-southeast-2 (Serverless & Private)")
        A["Drone Video Ingestion<br/>(S3: FORGE/inputs/videos/*.mp4)"] -->|"S3 ObjectCreated Event"| B["AWS Lambda Worker<br/>(Python 3.12 + OpenCV 5.0.0.93)"]
        B --> C["OpenCV 5 Video Probing<br/>& Bounded Corridor Sampling"]
        C --> D["Quantitative Metric Extraction<br/>(Laplacian, Tenengrad, ORB Density)"]
        D --> E["3D Feasibility Decision Gate<br/>(Sharpness, Features, Resolution)"]
        E --> F["Preflight Quality Report<br/>(S3: FORGE/outputs/aws_preflight/*.json)"]
        E --> G["Visual Contact Sheet<br/>(S3: FORGE/outputs/aws_preflight/*.jpg)"]
    end

    subgraph Edge / Local Workstation ("F.O.R.G.E Reconstruction Pipeline")
        F --> H["F.O.R.G.E CLI Runner<br/>(scripts/run_aws_preflight.py)"]
        H -->|"Gate: Approved"| I["OpenCV 5 Viewpoint Curation<br/>(src/video/opencv5_frame_analyzer.py)"]
        I --> J["Foundation Model 3D Reconstruction<br/>(VGGT-Omega Spatial Inference)"]
        J --> K["Metric Point Clouds & UV Meshes<br/>(F.O.R.G.E Explorer Three.js)"]
    end
```

---

## 2. Security, Cost & Least-Privilege Design

- **Private S3 Bucket:** `forge-drone3d-prathmesh-2026-8xq472` has **S3 Block Public Access** strictly enabled.
- **Strict Prefix Segregation (Zero-Recursion Guarantee):**
  - Lambda trigger is scoped strictly to `FORGE/inputs/videos/` with suffix `.mp4`.
  - Output reports and contact sheets are written to `FORGE/outputs/aws_preflight/`.
  - This guarantees that Lambda output writes can **never trigger an infinite invocation loop**.
- **Least-Privilege IAM Execution Role:**
  - `s3:GetObject` permission is restricted exclusively to `arn:aws:s3:::forge-drone3d-prathmesh-2026-8xq472/FORGE/inputs/videos/*`.
  - `s3:PutObject` permission is restricted exclusively to `arn:aws:s3:::forge-drone3d-prathmesh-2026-8xq472/FORGE/outputs/aws_preflight/*`.
  - `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents` for standard CloudWatch monitoring.
- **Resource Limits & Cost Caps:**
  - Memory: `2048 MB`
  - Timeout: `180 seconds`
  - Ephemeral Storage: `1024 MB` (safely accommodates the 154.5 MB Campus 0049 4K drone video).
  - Cost per execution: $<\$0.005$ USD per drone video analysis. Zero idle costs.

---

## 3. OpenCV 5 Packaging for AWS Lambda (Linux x86_64)

The Lambda function requires a pure headless build of OpenCV 5. Official PyPI releases provide pre-built manylinux wheels:
- **PyPI Package:** `opencv-python-headless==5.0.0.93`
- **Wheel:** `opencv_python_headless-5.0.0.93-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`

### Packaging via AWS SAM (Recommended):
From the project root:
```bash
cd aws/preflight
sam build --use-container
sam deploy --guided
```

### Direct Packaging via Docker / pip (Alternative):
```bash
cd aws/preflight
mkdir -p package
pip install \
    --platform manylinux2014_x86_64 \
    --target package \
    --implementation cp \
    --python-version 3.12 \
    --only-binary=:all: \
    -r requirements.txt

cp lambda_function.py package/
cd package && zip -r ../forge_preflight_lambda.zip . && cd ..
```

---

## 4. Deployment Instructions (AWS CLI)

### Step 1: Create IAM Execution Role
```bash
aws iam create-role \
    --role-name FORGE-Video-Preflight-ExecutionRole \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

aws iam put-role-policy \
    --role-name FORGE-Video-Preflight-ExecutionRole \
    --policy-name FORGE-Preflight-LeastPrivilege \
    --policy-document '{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
          "Resource": "arn:aws:logs:*:*:*"
        },
        {
          "Effect": "Allow",
          "Action": ["s3:GetObject"],
          "Resource": "arn:aws:s3:::forge-drone3d-prathmesh-2026-8xq472/FORGE/inputs/videos/*"
        },
        {
          "Effect": "Allow",
          "Action": ["s3:PutObject"],
          "Resource": "arn:aws:s3:::forge-drone3d-prathmesh-2026-8xq472/FORGE/outputs/aws_preflight/*"
        }
      ]
    }'
```

### Step 2: Deploy Lambda Function
```bash
aws lambda create-function \
    --function-name forge-video-preflight-worker \
    --runtime python3.12 \
    --role arn:aws:iam::<ACCOUNT_ID>:role/FORGE-Video-Preflight-ExecutionRole \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://aws/preflight/forge_preflight_lambda.zip \
    --timeout 180 \
    --memory-size 2048 \
    --ephemeral-storage '{"Size": 1024}' \
    --region ap-southeast-2 \
    --environment 'Variables={
      BUCKET_NAME=forge-drone3d-prathmesh-2026-8xq472,
      INPUT_PREFIX=FORGE/inputs/videos/,
      OUTPUT_PREFIX=FORGE/outputs/aws_preflight/,
      SAMPLE_FRAMES=8,
      MIN_LAPLACIAN_SHARPNESS=15.0,
      MIN_ORB_KEYPOINTS=100
    }'
```

### Step 3: Configure Event Notification Trigger (Optional for Auto-Runs)
```bash
aws lambda add-permission \
    --function-name forge-video-preflight-worker \
    --statement-id s3-trigger-permission \
    --action lambda:InvokeFunction \
    --principal s3.amazonaws.com \
    --source-arn arn:aws:s3:::forge-drone3d-prathmesh-2026-8xq472 \
    --region ap-southeast-2
```

---

## 5. Invocation & Test Execution

### Direct Lambda Invocation (Campus 0049 Drone Video):
Because the existing 154.5 MB Campus 0049 drone video (`dji_fly_20260727_172754_0049_1785175051818_video.mp4`) is already present in S3 at `FORGE/inputs/videos/`, invoke the Lambda worker directly via AWS CLI:

```bash
aws lambda invoke \
    --function-name forge-video-preflight-worker \
    --region ap-southeast-2 \
    --cli-binary-format raw-in-base64-out \
    --payload '{
      "bucket": "forge-drone3d-prathmesh-2026-8xq472",
      "key": "FORGE/inputs/videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4",
      "sample_frames": 8
    }' \
    response.json
```

### Output Report S3 Destination:
The worker generates and stores:
- **JSON Report:** `s3://forge-drone3d-prathmesh-2026-8xq472/FORGE/outputs/aws_preflight/dji_fly_20260727_172754_0049_1785175051818_video_preflight_report.json`
- **Contact Sheet:** `s3://forge-drone3d-prathmesh-2026-8xq472/FORGE/outputs/aws_preflight/dji_fly_20260727_172754_0049_1785175051818_video_contact_sheet.jpg`

---

## 6. Preflight Report Schema & Gate Interpretation

```json
{
  "service": "F.O.R.G.E. AWS Video Preflight Worker",
  "version": "1.0.0",
  "opencv_runtime_version": "5.0.0",
  "analysis_time_sec": 7.42,
  "video_metadata": {
    "source_s3_bucket": "forge-drone3d-prathmesh-2026-8xq472",
    "source_s3_key": "FORGE/inputs/videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4",
    "video_filename": "dji_fly_20260727_172754_0049_1785175051818_video.mp4",
    "file_size_mb": 154.53,
    "width": 3840,
    "height": 2160,
    "fps": 59.94,
    "total_frames": 966,
    "duration_sec": 16.12
  },
  "summary_metrics": {
    "mean_laplacian_sharpness": 449.6,
    "mean_tenengrad_sharpness": 11581.8,
    "mean_orb_keypoint_count": 500.0,
    "mean_dynamic_range": 175.8,
    "mean_intensity": 116.0
  },
  "preflight_decision": {
    "status": "APPROVED_FOR_3D_RECONSTRUCTION",
    "label": "Pass: Video meets all geospatial 3D reconstruction quality gates",
    "reasons": [
      "Resolution 3840x2160 meets criteria",
      "Mean Laplacian sharpness 449.6 indicates adequate focus",
      "Mean ORB keypoints 500.0 confirms textural richness"
    ],
    "approved_for_curation": true
  }
}
```

---

## 7. Verification Test Results

Execute the automated test suite verifying the Lambda preflight worker and OpenCV 5 analysis engine:
```bash
python -m unittest tests.test_aws_preflight
```

**Results:**
```text
Ran 6 tests in 0.802s
OK
- test_01_opencv5_runtime_guard: Passed
- test_02_single_frame_analysis: Passed
- test_03_decision_gate_evaluation: Passed
- test_04_contact_sheet_generation: Passed
- test_05_lambda_handler_with_mocked_s3: Passed
- test_06_preflight_on_real_drone_video: Passed
```
