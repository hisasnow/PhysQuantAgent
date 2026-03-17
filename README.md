# PhysQuantAgent
This repository provides a pipeline for estimating the physical weight of objects from RGB-D data captured via the Record3D app using Visual Language Models (VLMs). 
The system extracts real-world scale information using GroundingDINO, SAM, and Depth data, generates virtual cross-sectional blueprint images (Nano-banana) to infer internal structure/void ratio, and uses an agent-based VLM (GPT/Gemini) to perform the final weight inference.

## 🗂 Expected Directory Structure

```text
.
├── Visphysquant/
│   ├── convert_pose.py
│   ├── output.json
│   └── r3d_data/
│       ├── obj_001.r3d
│       ├── ...
├── GroundingDINO/                # Cloned GroundingDINO repository
│   ├── weights/                 
│       ├── groundingdino_swint_ogc.pth
│       └── sam_vit_h_4b8939.pth
├── Visual_prompt_images/          # Output directory for visual tool images
│   ├── bbox/                      # Annotated bounding box images
│   ├── axis/                      # Annotated scale estimation images
│   └── internal_image/            # Synthetic internal structure images
├── draw_bbox_axis.py
├── create_internal_image.py
└── physquantagent.py
```

## 🚀 Setup
1. Clone this repository.
2. Install the required Python packages (e.g., `opencv-python`, `numpy`, `scipy`, `google-generativeai`, `openai`, `torch`).
3. Set up [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything). Download and place their pre-trained weights (`groundingdino_swint_ogc.pth`, `sam_vit_h_4b8939.pth`) in the `weights/` directory.
4. Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API="your_openai_api_key"
   GEMINI_API="your_gemini_api_key"
   ```

## 🛠 Usage
### Dataset
First, download the dataset (VisPhysQuant) from the [link](https://drive.google.com/drive/folders/1_molSezAg9AwjlT4ZHmOuUwGbuB-wVYt?usp=drive_link)

The pipeline is designed to run in the following sequential steps:

### Step 1: Preprocess Record3D Data
Extract the RGB-D images and necessary camera metadata from the raw Record3D archives.
```bash
python Visphysquant/convert_pose.py
```

### Step 2: Object Detection & Scale Estimation 
Generate bounding boxes and measure the physical scale of the object.
```bash
python draw_bbox_axis.py --input-json Visphysquant/output.json --output-dir outputs
```

### Step 3: Generate Cross-Sectional Images
Generate the cross-sectional reference images (Nano-banana) using Gemini.
```bash
python create_internal_image.py
```

### Step 4: Run VLM Weight Inference and Evaluation
Run the multi-turn weight estimation oracle. You can switch between `gpt` and `gemini` by modifying the `API_PROVIDER` variable inside the script.
```bash
python physquantagent.py
```
After completion, the script will output the evaluation summary including overall success rate, MAPE, ADE, ALDE, and MNRE metrics.


## Citation

