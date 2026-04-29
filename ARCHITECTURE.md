# Deepfake Detection System — Complete Architecture Documentation

> Comprehensive technical architecture of the Deepfake Detection System, derived directly from the codebase. Every diagram maps to actual classes, functions, and data flows in the implementation.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Detailed Component Architecture](#2-detailed-component-architecture)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [Use Case Diagram](#4-use-case-diagram)
5. [Sequence Diagrams](#5-sequence-diagrams)
6. [Class Diagrams](#6-class-diagrams)
7. [Activity Diagrams](#7-activity-diagrams)
8. [System Component Interaction](#8-system-component-interaction)
9. [Deployment Architecture — Local Development](#9-deployment-architecture--local-development)
10. [Deployment Architecture — Production](#10-deployment-architecture--production)
11. [Security Architecture](#11-security-architecture)
12. [State Machine Diagram — Threat Score Zones](#12-state-machine-diagram--threat-score-zones)
13. [Entity Relationship Diagram](#13-entity-relationship-diagram)
14. [Architecture Summary](#14-architecture-summary)

---

## 1. System Architecture

> High-level overview showing every layer from raw media input through EfficientNet-B0 inference to final output. Maps directly to `web-app.py`, `classify.py`, and `inference/video_inference.py`.

```mermaid
graph TB
    subgraph "Layer 1 · Data Input"
        A["Raw Media Files<br/>.jpg .png .mp4 .mov"] --> B["mimetypes.guess_type()<br/>web-app.py:33"]
        B --> C{Media Type?}
        C -->|"image/*"| D["PIL.Image.open()<br/>→ RGB convert"]
        C -->|"video/*"| E["cv2.VideoCapture()<br/>Frame Extraction"]
        C -->|"unsupported"| F["⚠ Error Response"]
    end

    subgraph "Layer 2 · Preprocessing Pipeline"
        D --> G["transforms.Resize(224,224)"]
        E --> H["Frame Sampling<br/>np.linspace N frames"]
        H --> G
        G --> I["transforms.ToTensor()"]
        I --> J["transforms.Normalize<br/>μ=[0.485,0.456,0.406]<br/>σ=[0.229,0.224,0.225]"]
        J --> K[".unsqueeze(0)<br/>Batch dim [1,3,224,224]"]
    end

    subgraph "Layer 3 · Feature Extraction"
        K --> L["EfficientNet-B0<br/>ImageNet pretrained"]
        L --> M["Conv Stem → MBConv Blocks"]
        M --> N["Adaptive Avg Pool<br/>→ 1280-D vector"]
    end

    subgraph "Layer 4 · Classification Head"
        N --> O["nn.Dropout(p=0.4)"]
        O --> P["nn.Linear(1280 → 2)"]
        P --> Q["torch.softmax(dim=1)"]
    end

    subgraph "Layer 5 · Post-Processing"
        Q --> R["torch.max(probs)<br/>conf, pred"]
        R --> S{"pred == 0?"}
        S -->|Yes| T["🟢 Real"]
        S -->|No| U["🔴 Deepfake"]
        T --> V["Confidence %"]
        U --> V
    end

    subgraph "Layer 6 · Output"
        V --> W["Gradio Web UI<br/>gr.Blocks"]
        V --> X["CLI stdout<br/>classify.py"]
        V --> Y["Batch folder scan<br/>realeval.py"]
    end

    style L fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
    style S fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:3px
    style W fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
```

---

## 2. Detailed Component Architecture

> Maps each module in the repository to its role in the Training, Inference, Video Processing, and Tooling pipelines.

```mermaid
graph LR
    subgraph "Training Pipeline"
        A1["config.yaml<br/>lr, batch_size, epochs"] --> B1["main_trainer.py"]
        B1 --> C1["HybridDeepfakeDataset<br/>datasets/hybrid_loader.py"]
        C1 --> D1["DataLoader<br/>batch_size=4, shuffle"]
        D1 --> E1["DeepfakeDetector<br/>lightning_modules/detector.py"]
        E1 --> F1["pl.Trainer<br/>GPU/CPU auto-detect"]
        F1 --> G1["ModelCheckpoint<br/>→ models/best_model.ckpt"]
        F1 --> H1["EarlyStopping<br/>patience=3"]
    end

    subgraph "Inference Pipeline"
        A2["User Input<br/>image path"] --> B2["classify.py<br/>argparse CLI"]
        B2 --> C2["load_model()<br/>best_model.pt"]
        C2 --> D2["predict_image()<br/>forward pass"]
        D2 --> E2["Print REAL/FAKE<br/>+ probabilities"]
    end

    subgraph "Video Inference"
        A3["Video Folder<br/>videos_to_predict/"] --> B3["video_inference.py"]
        B3 --> C3["extract_frames()<br/>N=10 via linspace"]
        C3 --> D3["Per-frame predict<br/>softmax probs"]
        D3 --> E3["torch.mean(stack)<br/>Avg probability"]
        E3 --> F3["argmax → label"]
    end

    subgraph "Web Application"
        A4["Gradio gr.Blocks<br/>web-app.py"] --> B4["gr.File upload<br/>.jpg .png .mp4 .mov"]
        B4 --> C4["predict_file()<br/>MIME routing"]
        C4 --> D4["gr.Textbox Prediction<br/>gr.Textbox Confidence<br/>gr.Image Preview"]
    end

    subgraph "Utility Tools"
        T1["split_dataset.py<br/>Video→Frame extraction"]
        T2["split_train_val.py<br/>80/20 train/val split"]
        T3["split_video_dataset.py<br/>Video-aware splitting"]
        T4["export_to_pt.py<br/>.ckpt → .pt conversion"]
        T5["export_onnx.py<br/>.pt → .onnx export"]
    end

    style E1 fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
    style C2 fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:3px
    style A4 fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
```

---

## 3. Data Flow Architecture

> Traces data transformation from raw upload through every processing stage. Values reference actual code parameters.

```mermaid
flowchart TD
    subgraph "Stage 1 · Input Validation"
        A["User Upload<br/>via Gradio / CLI"] --> B{"file_obj is None?"}
        B -->|Yes| C["⚠ No file selected<br/>return early"]
        B -->|No| D["path = file_obj.name"]
        D --> E["mime = mimetypes.guess_type(path)"]
        E --> F{"mime.startswith?"}
    end

    subgraph "Stage 2 · Media Loading"
        F -->|"image"| G["Image.open(path)<br/>.convert('RGB')"]
        F -->|"video"| H["cv2.VideoCapture(path)<br/>cap.read() → frame"]
        F -->|"other"| I["Unsupported file type"]
        H --> J["cvtColor BGR→RGB<br/>Image.fromarray()"]
        J --> G2["PIL Image ready"]
        G --> G2
    end

    subgraph "Stage 3 · Tensor Pipeline"
        G2 --> K["Resize(224, 224)<br/>Bilinear interpolation"]
        K --> L["ToTensor()<br/>[H,W,C] → [C,H,W]<br/>uint8 → float32 /255"]
        L --> M["Normalize<br/>μ=[0.485,0.456,0.406]<br/>σ=[0.229,0.224,0.225]"]
        M --> N[".unsqueeze(0)<br/>[3,224,224]→[1,3,224,224]"]
    end

    subgraph "Stage 4 · Model Forward Pass"
        N --> O["torch.no_grad()"]
        O --> P["model(tensor)<br/>→ logits [1, 2]"]
        P --> Q["softmax(dim=1)<br/>→ probs [1, 2]"]
        Q --> R["torch.max(probs, dim=0)<br/>→ conf, pred"]
    end

    subgraph "Stage 5 · Decision & Output"
        R --> S{"pred.item()"}
        S -->|"0"| T["🟢 Real"]
        S -->|"1"| U["🔴 Deepfake"]
        T --> V["conf.item()*100<br/>→ Confidence %"]
        U --> V
        V --> W["Return: label, confidence, preview_img"]
    end

    style P fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
    style S fill:#f59e0b,stroke:#d97706,color:#fff,stroke-width:3px
    style V fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
```

---

## 4. Use Case Diagram

> All user-facing capabilities of the system, derived from entry points: `web-app.py`, `classify.py`, `realeval.py`, `main_trainer.py`, and tool scripts.

```mermaid
graph TB
    User["👤 End User"]
    Dev["👨‍💻 ML Developer"]
    
    subgraph "Web Interface Use Cases"
        UC1["Upload Image for Detection"]
        UC2["Upload Video for Detection"]
        UC3["View Prediction & Confidence"]
        UC4["Preview Analyzed Frame"]
    end

    subgraph "CLI Use Cases"
        UC5["Classify Single Image<br/>classify.py image_path"]
        UC6["Batch Evaluate Folder<br/>realeval.py"]
        UC7["Evaluate with Noise Simulation<br/>realeval.py simulate=True"]
        UC8["Run Video Inference<br/>inference/video_inference.py"]
    end

    subgraph "Developer / Training Use Cases"
        UC9["Train Model<br/>main_trainer.py"]
        UC10["Configure Hyperparameters<br/>config.yaml"]
        UC11["Split Image Dataset<br/>tools/split_train_val.py"]
        UC12["Extract Frames from Video<br/>tools/split_dataset.py"]
        UC13["Split Video Dataset<br/>tools/split_video_dataset.py"]
        UC14["Export .ckpt → .pt<br/>tools/export_to_pt.py"]
        UC15["Export .pt → .onnx<br/>inference/export_onnx.py"]
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6

    Dev --> UC9
    Dev --> UC10
    Dev --> UC11
    Dev --> UC12
    Dev --> UC13
    Dev --> UC14
    Dev --> UC15
    Dev --> UC7
    Dev --> UC8

    style User fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:2px
    style Dev fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:2px
```

---

## 5. Sequence Diagrams

### 5.1 Web App — Image Upload Flow (`web-app.py`)

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio UI<br/>gr.Blocks
    participant Handler as handle_input()
    participant Predict as predict_file()
    participant MIME as mimetypes
    participant PIL as PIL.Image
    participant Transform as torchvision.transforms
    participant Model as EfficientNet-B0
    participant Softmax as torch.softmax

    User->>UI: Drop image file (.jpg/.png)
    UI->>Handler: file_input.change event
    Handler->>Predict: predict_file(file_obj)
    Predict->>Predict: path = file_obj.name
    Predict->>MIME: guess_type(path)
    MIME-->>Predict: ("image/jpeg", None)
    Predict->>PIL: Image.open(path).convert("RGB")
    PIL-->>Predict: PIL Image
    Predict->>Transform: preprocess(img).unsqueeze(0)
    Note over Transform: Resize(224) → ToTensor → Normalize
    Transform-->>Predict: tensor [1,3,224,224]
    Predict->>Model: model(tensor) [torch.no_grad]
    Model-->>Predict: logits [1, 2]
    Predict->>Softmax: softmax(logits, dim=1)[0]
    Softmax-->>Predict: probs [2]
    Predict->>Predict: conf, pred = torch.max(probs)
    Predict-->>Handler: (label, confidence%, preview_img)
    Handler-->>UI: Update outputs
    UI-->>User: Show 🟢Real/🔴Deepfake + Confidence + Preview
```

### 5.2 Web App — Video Upload Flow (`web-app.py`)

```mermaid
sequenceDiagram
    actor User
    participant UI as Gradio UI
    participant Predict as predict_file()
    participant CV2 as OpenCV
    participant PIL as PIL.Image
    participant Model as EfficientNet-B0

    User->>UI: Drop video file (.mp4/.mov)
    UI->>Predict: predict_file(file_obj)
    Predict->>Predict: mime = "video/mp4"
    Predict->>CV2: VideoCapture(path)
    CV2->>CV2: cap.read() → 1st frame only
    CV2-->>Predict: ret, frame (BGR)
    alt ret == False
        Predict-->>UI: "❌ Error reading video"
    end
    Predict->>CV2: cvtColor(BGR → RGB)
    Predict->>PIL: Image.fromarray(frame)
    PIL-->>Predict: PIL Image
    Predict->>Model: preprocess → model(tensor)
    Model-->>Predict: probs
    Predict-->>UI: "🟢 Real (1st frame)" or "🔴 Deepfake (1st frame)"
```

### 5.3 Multi-Frame Video Inference (`inference/video_inference.py`)

```mermaid
sequenceDiagram
    participant Script as video_inference.py
    participant Extract as extract_frames()
    participant CV2 as OpenCV
    participant Model as EfficientNet-B0
    participant Agg as Probability Aggregator

    Script->>Script: Scan videos_to_predict/ folder
    loop Each .mp4 file
        Script->>Extract: extract_frames(path, num_frames=10)
        Extract->>CV2: VideoCapture(path)
        Extract->>CV2: total = CAP_PROP_FRAME_COUNT
        Extract->>Extract: indexes = np.linspace(0, total-1, 10)
        loop Each frame index
            CV2-->>Extract: frame BGR
            Extract->>Extract: cvtColor → PIL Image
        end
        Extract-->>Script: List[PIL.Image] (10 frames)
        
        loop Each frame (torch.no_grad)
            Script->>Model: transform(frame).unsqueeze(0)
            Model-->>Script: softmax probs [1, 2]
            Script->>Agg: append prob
        end
        Script->>Agg: torch.mean(torch.stack(all_probs), dim=0)
        Agg-->>Script: avg_prob
        Script->>Script: argmax → REAL/FAKE
        Script->>Script: print result
    end
```

### 5.4 Training Pipeline (`main_trainer.py`)

```mermaid
sequenceDiagram
    participant Config as config.yaml
    participant Trainer as main_trainer.py
    participant Dataset as HybridDeepfakeDataset
    participant Loader as DataLoader
    participant Lightning as DeepfakeDetector
    participant PLTrainer as pl.Trainer
    participant Callbacks as Checkpoint + EarlyStopping

    Trainer->>Config: yaml.safe_load()
    Config-->>Trainer: cfg dict
    Trainer->>Dataset: HybridDeepfakeDataset(train_sources, transform)
    Dataset->>Dataset: Walk real/ and fake/ subdirs
    Dataset->>Dataset: Collect .jpg/.png paths + labels
    Trainer->>Loader: DataLoader(batch_size=4, shuffle=True)
    Trainer->>Trainer: Build EfficientNet-B0 backbone
    Note over Trainer: classifier = Dropout(0.4) + Linear(1280→2)
    Trainer->>Lightning: DeepfakeDetector(backbone, lr=0.0001)
    Trainer->>Callbacks: ModelCheckpoint(monitor=val_loss, save_top_k=1)
    Trainer->>Callbacks: EarlyStopping(patience=3)
    Trainer->>PLTrainer: pl.Trainer(max_epochs, accelerator, callbacks)
    PLTrainer->>PLTrainer: trainer.fit(model, train_loader, val_loader)
    
    loop Each Epoch
        loop Each Batch
            PLTrainer->>Lightning: training_step(batch)
            Lightning->>Lightning: logits = model(x)
            Lightning->>Lightning: loss = CrossEntropyLoss(logits, y)
            Lightning->>Lightning: log train_loss, train_acc
        end
        loop Validation Batches
            PLTrainer->>Lightning: validation_step(batch)
            Lightning->>Lightning: log val_loss, val_acc
        end
        PLTrainer->>Callbacks: Check val_loss improvement
        alt No improvement for 3 epochs
            Callbacks-->>PLTrainer: Stop training
        end
    end
    PLTrainer->>Callbacks: Save best_model.ckpt
```

---

## 6. Class Diagrams

> Actual classes, methods, attributes, and inheritance from the codebase.

```mermaid
classDiagram
    class DeepfakeDetector {
        <<LightningModule>>
        +model: nn.Module
        +lr: float
        +loss_fn: CrossEntropyLoss
        +__init__(model, lr=1e-4)
        +forward(x) Tensor
        +training_step(batch, batch_idx) Tensor
        +validation_step(batch, batch_idx) void
        +configure_optimizers() Adam
    }

    class HybridDeepfakeDataset {
        <<Dataset>>
        +transform: Compose
        +image_paths: List~str~
        +labels: List~int~
        -class_map: dict = real:0 fake:1
        +__init__(sources, transform)
        +__len__() int
        +__getitem__(idx) Tuple~Tensor,int~
    }

    class EfficientNetB0Backbone {
        <<torchvision>>
        +features: Sequential
        +avgpool: AdaptiveAvgPool2d
        +classifier: Sequential
        +forward(x) Tensor
    }

    class ClassifierHead {
        <<nn.Sequential>>
        +dropout: Dropout(p=0.4)
        +linear: Linear(1280, 2)
    }

    class TransformPipeline {
        <<torchvision.transforms.Compose>>
        +Resize(224, 224)
        +ToTensor()
        +Normalize(mean, std)
    }

    class GradioWebApp {
        <<web-app.py>>
        +model: EfficientNetB0
        +preprocess: Compose
        +load_model() nn.Module
        +predict_file(file_obj) Tuple
        +handle_input(file_obj) Tuple
        +demo: gr.Blocks
    }

    class CLIClassifier {
        <<classify.py>>
        +load_model(path) nn.Module
        +predict_image(path, model) void
    }

    class RealWorldEvaluator {
        <<realeval.py>>
        +model: EfficientNetB0
        +distort(image, simulate) Tensor
        +evaluate(folder, simulate_noise) void
    }

    class VideoInference {
        <<video_inference.py>>
        +model: EfficientNetB0
        +device: torch.device
        +transform: Compose
        +extract_frames(path, num_frames) List
        +predict_video(path) Tuple
    }

    DeepfakeDetector *-- EfficientNetB0Backbone : wraps
    EfficientNetB0Backbone *-- ClassifierHead : classifier layer
    HybridDeepfakeDataset ..> TransformPipeline : uses
    GradioWebApp ..> EfficientNetB0Backbone : loads
    GradioWebApp ..> TransformPipeline : uses
    CLIClassifier ..> EfficientNetB0Backbone : loads
    CLIClassifier ..> TransformPipeline : uses
    RealWorldEvaluator ..> EfficientNetB0Backbone : loads
    VideoInference ..> EfficientNetB0Backbone : loads
    VideoInference ..> TransformPipeline : uses
```

### 6.2 Tools & Export Classes

```mermaid
classDiagram
    class ExportToPt {
        <<tools/export_to_pt.py>>
        +ckpt_path: str
        +pt_output: str
        +Load .ckpt via LightningModule
        +Save model.state_dict() as .pt
    }

    class ExportONNX {
        <<inference/export_onnx.py>>
        +Load .pt model
        +dummy_input: randn(1,3,224,224)
        +torch.onnx.export()
        +Output: deepfake_model.onnx
    }

    class SplitDataset {
        <<tools/split_dataset.py>>
        +extract_frames_from_video(path, out_dir, every_n=15)
        +Read video → save every Nth frame as .jpg
    }

    class SplitTrainVal {
        <<tools/split_train_val.py>>
        +split_dataset(source, dest, ratio=0.8)
        +Shuffle → 80% train / 20% val
        +Copy files to train/real, train/fake, etc.
    }

    class SplitVideoDataset {
        <<tools/split_video_dataset.py>>
        +extract_and_split_videos(source, dest, ratio=0.8)
        +Split videos first, then extract frames
        +frames_per_video=5, every_n_frames=15
    }

    ExportToPt ..> DeepfakeDetector : loads checkpoint
    ExportONNX ..> EfficientNetB0Backbone : exports
    SplitDataset ..> HybridDeepfakeDataset : prepares data for
    SplitTrainVal ..> HybridDeepfakeDataset : prepares data for
    SplitVideoDataset ..> HybridDeepfakeDataset : prepares data for
```

---

## 7. Activity Diagrams

### 7.1 Image Classification Activity

```mermaid
flowchart TD
    Start((●)) --> A["Receive file input"]
    A --> B{"file_obj is None?"}
    B -->|Yes| C["Return: ⚠ No file selected"]
    C --> End1((◉))
    B -->|No| D["Extract file path"]
    D --> E["Detect MIME type"]
    E --> F{"Is image?"}
    F -->|Yes| G["PIL.Image.open → RGB"]
    F -->|No| H{"Is video?"}
    H -->|Yes| I["VideoCapture → read 1st frame"]
    I --> J{"Frame read OK?"}
    J -->|No| K["Return: ❌ Error"]
    K --> End2((◉))
    J -->|Yes| L["cvtColor BGR→RGB → PIL Image"]
    L --> G
    H -->|No| M["Return: Unsupported type"]
    M --> End3((◉))
    G --> N["Resize to 224×224"]
    N --> O["ToTensor + Normalize"]
    O --> P["unsqueeze → batch dim"]
    P --> Q["torch.no_grad context"]
    Q --> R["model(tensor) → logits"]
    R --> S["softmax → probabilities"]
    S --> T["torch.max → conf, pred"]
    T --> U{"pred == 0?"}
    U -->|Yes| V["label = 🟢 Real"]
    U -->|No| W["label = 🔴 Deepfake"]
    V --> X["Format confidence %"]
    W --> X
    X --> Y["Return label, confidence, preview"]
    Y --> End4((◉))

    style Start fill:#22c55e,stroke:#16a34a,color:#fff
    style End1 fill:#ef4444,stroke:#dc2626,color:#fff
    style End2 fill:#ef4444,stroke:#dc2626,color:#fff
    style End3 fill:#ef4444,stroke:#dc2626,color:#fff
    style End4 fill:#22c55e,stroke:#16a34a,color:#fff
```

### 7.2 Model Training Activity

```mermaid
flowchart TD
    Start((●)) --> A["Load config.yaml"]
    A --> B["Define transforms<br/>Resize, ToTensor, Normalize"]
    B --> C["Build train_sources & val_sources<br/>from config paths"]
    C --> D["Create HybridDeepfakeDataset<br/>Walk real/ & fake/ subdirs"]
    D --> E["Create DataLoaders<br/>batch_size=4"]
    E --> F["Initialize EfficientNet-B0<br/>ImageNet pretrained weights"]
    F --> G["Replace classifier head<br/>Dropout(0.4) + Linear(1280→2)"]
    G --> H["Wrap in DeepfakeDetector<br/>LightningModule"]
    H --> I["Configure callbacks<br/>ModelCheckpoint + EarlyStopping"]
    I --> J["Create pl.Trainer<br/>auto GPU/CPU"]
    J --> K["trainer.fit()"]
    K --> L{"Epoch loop"}
    L --> M["training_step: forward → CE loss → backprop"]
    M --> N["Log train_loss, train_acc"]
    N --> O["validation_step: forward → CE loss"]
    O --> P["Log val_loss, val_acc"]
    P --> Q{"val_loss improved?"}
    Q -->|Yes| R["Save checkpoint<br/>models/best_model.ckpt"]
    R --> L
    Q -->|No| S["Increment patience counter"]
    S --> T{"patience >= 3?"}
    T -->|No| L
    T -->|Yes| U["Early stopping triggered"]
    U --> V["Training complete"]
    V --> End((◉))

    style Start fill:#a855f7,stroke:#7c3aed,color:#fff
    style End fill:#22c55e,stroke:#16a34a,color:#fff
    style U fill:#f59e0b,stroke:#d97706,color:#fff
```

### 7.3 Dataset Preparation Activity

```mermaid
flowchart TD
    Start((●)) --> A{"Input type?"}
    A -->|"Raw videos<br/>in real/ & fake/"| B["split_video_dataset.py"]
    A -->|"Already extracted<br/>images"| C["split_train_val.py"]
    A -->|"Videos needing<br/>frame extraction only"| D["split_dataset.py"]

    B --> E["List .mp4 files per label"]
    E --> F["random.shuffle()"]
    F --> G["Split 80/20 by video"]
    G --> H["Extract 5 frames per video<br/>every 15th frame"]
    H --> I["Save to dest/train/label/<br/>& dest/validation/label/"]

    C --> J["List .jpg/.png per label"]
    J --> K["random.shuffle()"]
    K --> L["Split 80/20 by file"]
    L --> M["shutil.copy to<br/>train/ & validation/"]

    D --> N["cv2.VideoCapture per .mp4"]
    N --> O["Save every 15th frame<br/>as .jpg to output_dir"]

    I --> End((◉))
    M --> End
    O --> End

    style Start fill:#3b82f6,stroke:#2563eb,color:#fff
    style End fill:#22c55e,stroke:#16a34a,color:#fff
```

---

## 8. System Component Interaction

> Shows how every file in the repository interacts with every other file at runtime and build time.

```mermaid
graph TB
    subgraph "Configuration"
        CFG["config.yaml<br/>lr, batch_size, epochs<br/>train_paths, val_paths"]
    end

    subgraph "Data Layer"
        HL["datasets/hybrid_loader.py<br/>HybridDeepfakeDataset"]
        SD["tools/split_dataset.py<br/>Video → Frames"]
        STV["tools/split_train_val.py<br/>80/20 split"]
        SVD["tools/split_video_dataset.py<br/>Video-aware split"]
    end

    subgraph "Model Layer"
        DET["lightning_modules/detector.py<br/>DeepfakeDetector (LightningModule)"]
        ENET["EfficientNet-B0<br/>(torchvision)"]
        WEIGHTS["models/best_model-v3.pt<br/>Saved weights (15.6 MB)"]
    end

    subgraph "Training"
        MT["main_trainer.py<br/>pl.Trainer orchestrator"]
    end

    subgraph "Inference Interfaces"
        WA["web-app.py<br/>Gradio UI"]
        CL["classify.py<br/>CLI single image"]
        RE["realeval.py<br/>Batch eval + noise sim"]
        VI["inference/video_inference.py<br/>Multi-frame video"]
    end

    subgraph "Export"
        EPT["tools/export_to_pt.py<br/>.ckpt → .pt"]
        EON["inference/export_onnx.py<br/>.pt → .onnx"]
    end

    CFG --> MT
    MT --> HL
    MT --> DET
    DET --> ENET
    MT -->|"saves"| WEIGHTS

    SD -->|"prepares data for"| HL
    STV -->|"prepares data for"| HL
    SVD -->|"prepares data for"| HL

    WEIGHTS --> WA
    WEIGHTS --> CL
    WEIGHTS --> RE
    WEIGHTS --> VI

    WA --> ENET
    CL --> ENET
    RE --> ENET
    VI --> ENET

    EPT -->|"reads .ckpt"| DET
    EPT -->|"writes .pt"| WEIGHTS
    EON -->|"reads .pt"| WEIGHTS

    style ENET fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
    style WEIGHTS fill:#f59e0b,stroke:#d97706,color:#fff,stroke-width:3px
    style WA fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
```

---

## 9. Deployment Architecture — Local Development

> Actual local setup: single-machine, no containers, Python virtual environment.

```mermaid
graph TB
    subgraph "Developer Machine"
        subgraph "Python Environment"
            PY["Python 3.x<br/>Virtual Environment"]
            DEPS["requirements.txt<br/>torch, torchvision, pytorch-lightning<br/>gradio, opencv-python, PyYAML<br/>Pillow, numpy, scikit-learn, tensorboard"]
        end

        subgraph "Application Process"
            WA["web-app.py<br/>python web-app.py"]
            GRADIO["Gradio Server<br/>http://127.0.0.1:7860"]
        end

        subgraph "File System"
            MODEL["models/<br/>best_model-v3.pt (15.6 MB)"]
            DATA["datasets/<br/>hybrid_loader.py"]
            TOOLS["tools/<br/>split & export scripts"]
            INF["inference/<br/>video & onnx export"]
            LM["lightning_modules/<br/>detector.py"]
            CONFIG["config.yaml"]
        end

        subgraph "GPU / CPU"
            CUDA{"CUDA Available?"}
            GPU["NVIDIA GPU<br/>CUDA Acceleration"]
            CPU["CPU Fallback"]
        end
    end

    subgraph "External"
        BROWSER["🌐 Web Browser<br/>localhost:7860"]
        TERMINAL["💻 Terminal<br/>CLI commands"]
        DATASET["📁 External Dataset<br/>C:/Datasets/..."]
    end

    PY --> DEPS
    WA --> GRADIO
    GRADIO --> MODEL
    CUDA -->|Yes| GPU
    CUDA -->|No| CPU
    BROWSER --> GRADIO
    TERMINAL --> WA
    TERMINAL --> INF
    CONFIG --> DATA
    DATASET --> DATA

    style GRADIO fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
    style MODEL fill:#f59e0b,stroke:#d97706,color:#fff,stroke-width:3px
    style GPU fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
```

---

## 10. Deployment Architecture — Production

> Recommended production deployment with containerization, reverse proxy, and model serving.

```mermaid
graph TB
    subgraph "Internet"
        USER["👤 End Users"]
    end

    subgraph "Edge / CDN"
        CF["Cloudflare / CDN<br/>DDoS Protection<br/>SSL Termination"]
    end

    subgraph "Reverse Proxy"
        NGINX["Nginx<br/>Rate Limiting<br/>Request Routing"]
    end

    subgraph "Application Container"
        subgraph "Docker Container"
            PY["Python Runtime"]
            APP["web-app.py<br/>Gradio Server :7860"]
            MODEL["EfficientNet-B0<br/>best_model-v3.pt"]
        end
    end

    subgraph "GPU Server"
        CUDA["CUDA Runtime"]
        GPU["NVIDIA GPU<br/>Inference Acceleration"]
    end

    subgraph "Model Registry"
        S3["S3 / GCS Bucket<br/>Model Versioning<br/>v1, v2, v3"]
    end

    subgraph "Monitoring"
        TB["TensorBoard<br/>Training Metrics"]
        LOGS["Application Logs<br/>Predictions & Errors"]
    end

    subgraph "CI/CD"
        GH["GitHub Actions"]
        BUILD["Docker Build<br/>+ Push"]
        DEPLOY["Deploy to Server"]
    end

    USER --> CF
    CF --> NGINX
    NGINX --> APP
    APP --> MODEL
    MODEL --> GPU
    GPU --> CUDA
    S3 -->|"model pull"| MODEL
    APP --> LOGS
    GH --> BUILD
    BUILD --> DEPLOY
    DEPLOY --> APP

    style APP fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:3px
    style GPU fill:#a855f7,stroke:#7c3aed,color:#fff,stroke-width:3px
    style CF fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:2px
```

---

## 11. Security Architecture

> File validation, model integrity, and input sanitization as implemented in the codebase.

```mermaid
graph TB
    subgraph "Input Security"
        A["User Upload"] --> B["Gradio File Filter<br/>.jpg .jpeg .png .mp4 .mov"]
        B --> C["MIME Type Validation<br/>mimetypes.guess_type()"]
        C --> D{"Valid MIME?"}
        D -->|No| E["❌ Reject<br/>Unsupported file type"]
        D -->|Yes| F["✅ Accept for processing"]
    end

    subgraph "Processing Security"
        F --> G["torch.no_grad()<br/>No gradient computation"]
        G --> H["Fixed input size<br/>Resize(224,224)<br/>Prevents memory attacks"]
        H --> I["Single file processing<br/>No batch injection"]
    end

    subgraph "Model Security"
        J["Model weights .pt file<br/>Local filesystem only"]
        K["map_location='cpu'<br/>Safe deserialization"]
        L["model.eval() mode<br/>Dropout disabled"]
    end

    subgraph "Output Security"
        M["Softmax bounded [0,1]<br/>No raw logit exposure"]
        N["Confidence as percentage<br/>Human-readable output"]
        O["No PII stored<br/>Uploads are temporary"]
    end

    subgraph "Network Security"
        P["Gradio local server<br/>127.0.0.1:7860 default"]
        Q["No external API calls<br/>Fully offline capable"]
    end

    F --> J
    J --> K
    K --> L
    L --> G
    I --> M
    M --> N
    N --> O

    style E fill:#ef4444,stroke:#dc2626,color:#fff,stroke-width:2px
    style F fill:#22c55e,stroke:#16a34a,color:#fff,stroke-width:2px
    style Q fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:2px
```

---

## 12. State Machine Diagram — Threat Score Zones

> Maps the softmax probability output to threat assessment zones.

```mermaid
stateDiagram-v2
    [*] --> FileReceived

    FileReceived --> Validating : Upload detected
    Validating --> Rejected : Invalid MIME type
    Validating --> Processing : Valid file

    Rejected --> [*]

    Processing --> Inferencing : Tensor ready [1,3,224,224]
    Inferencing --> Scoring : Softmax probs computed

    Scoring --> SafeZone : fake_prob < 0.30
    Scoring --> CautionZone : 0.30 ≤ fake_prob < 0.50
    Scoring --> WarningZone : 0.50 ≤ fake_prob < 0.70
    Scoring --> DangerZone : 0.70 ≤ fake_prob < 0.90
    Scoring --> CriticalZone : fake_prob ≥ 0.90

    state SafeZone {
        [*] --> DisplayReal
        DisplayReal : 🟢 REAL
        DisplayReal : Confidence > 70%
        DisplayReal : Threat Level: SAFE
    }

    state CautionZone {
        [*] --> DisplayCaution
        DisplayCaution : 🟡 LIKELY REAL
        DisplayCaution : Low manipulation indicators
        DisplayCaution : Threat Level: LOW
    }

    state WarningZone {
        [*] --> DisplayWarning
        DisplayWarning : 🟠 SUSPICIOUS
        DisplayWarning : Moderate manipulation signals
        DisplayWarning : Threat Level: MEDIUM
    }

    state DangerZone {
        [*] --> DisplayDanger
        DisplayDanger : 🔴 LIKELY DEEPFAKE
        DisplayDanger : Strong manipulation detected
        DisplayDanger : Threat Level: HIGH
    }

    state CriticalZone {
        [*] --> DisplayCritical
        DisplayCritical : 🔴⚠ DEEPFAKE
        DisplayCritical : Very high confidence
        DisplayCritical : Threat Level: CRITICAL
    }

    SafeZone --> [*]
    CautionZone --> [*]
    WarningZone --> [*]
    DangerZone --> [*]
    CriticalZone --> [*]
```

---

## 13. Entity Relationship Diagram

> Data entities and their relationships across the system.

```mermaid
erDiagram
    CONFIG {
        float lr "0.0001"
        int batch_size "4"
        int num_epochs "1"
        string train_paths "list of dirs"
        string val_paths "list of dirs"
        int early_stopping_patience "3"
        float scheduler_factor "0.5"
        int scheduler_patience "2"
        string monitor_metric "val_loss"
        int log_every_n_steps "1"
        int save_top_k "1"
    }

    DATASET {
        string image_path "absolute path to .jpg/.png"
        int label "0=real, 1=fake"
        string source_dir "real/ or fake/ subdir"
    }

    MODEL {
        string architecture "EfficientNet-B0"
        int input_channels "3 (RGB)"
        int input_height "224"
        int input_width "224"
        int feature_dim "1280"
        float dropout_rate "0.4"
        int num_classes "2"
        string pretrained "ImageNet1K_V1"
    }

    CHECKPOINT {
        string filepath "models/best_model.ckpt"
        float val_loss "monitored metric"
        int epoch "training epoch"
        string optimizer_state "Adam state dict"
    }

    WEIGHTS {
        string filepath "models/best_model-v3.pt"
        int size_bytes "16341059"
        string format "state_dict"
    }

    PREDICTION {
        string input_path "uploaded file path"
        string mime_type "image/* or video/*"
        string label "Real or Deepfake"
        float confidence "0.0 to 100.0"
        float prob_real "softmax[0]"
        float prob_fake "softmax[1]"
    }

    VIDEO_ANALYSIS {
        string video_path "path to .mp4/.mov"
        int num_frames "10 sampled frames"
        float avg_prob_real "mean of frame probs"
        float avg_prob_fake "mean of frame probs"
        string final_label "REAL or FAKE"
    }

    TRANSFORM {
        string resize "224x224"
        string to_tensor "HWC uint8 → CHW float32"
        string normalize_mean "0.485, 0.456, 0.406"
        string normalize_std "0.229, 0.224, 0.225"
    }

    CONFIG ||--o{ DATASET : "defines paths for"
    CONFIG ||--|| MODEL : "configures hyperparams"
    DATASET }o--|| TRANSFORM : "preprocessed by"
    MODEL ||--|| CHECKPOINT : "saved as"
    CHECKPOINT ||--|| WEIGHTS : "exported to"
    WEIGHTS ||--o{ PREDICTION : "generates"
    WEIGHTS ||--o{ VIDEO_ANALYSIS : "generates"
    VIDEO_ANALYSIS }o--|| PREDICTION : "aggregates into"
    TRANSFORM ||--o{ PREDICTION : "applied before"
```

---

## 14. Architecture Summary

### Repository Structure Map

```
DeepfakeDetector/
├── config.yaml                    # Hyperparameters & dataset paths
├── main_trainer.py                # Training orchestrator (PyTorch Lightning)
├── classify.py                    # CLI single-image classifier
├── realeval.py                    # Batch evaluator with noise simulation
├── web-app.py                     # Gradio web interface
├── requirements.txt               # Python dependencies (10 packages)
│
├── models/
│   └── best_model-v3.pt           # Trained weights (15.6 MB)
│
├── datasets/
│   └── hybrid_loader.py           # HybridDeepfakeDataset (PyTorch Dataset)
│
├── lightning_modules/
│   └── detector.py                # DeepfakeDetector (LightningModule)
│
├── inference/
│   ├── video_inference.py         # Multi-frame video classification
│   └── export_onnx.py             # ONNX model export
│
└── tools/
    ├── export_to_pt.py            # .ckpt → .pt converter
    ├── split_dataset.py           # Video frame extractor
    ├── split_train_val.py         # Image dataset splitter (80/20)
    └── split_video_dataset.py     # Video-aware dataset splitter
```

### Core Technical Specifications

| Component | Specification |
|---|---|
| **Backbone** | EfficientNet-B0 (torchvision, ImageNet-1K pretrained) |
| **Feature Dimension** | 1280-D (Global Average Pooling output) |
| **Classifier Head** | Dropout(0.4) → Linear(1280, 2) |
| **Input Resolution** | 224 × 224 × 3 (RGB) |
| **Normalization** | μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225] |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | Adam (lr=0.0001) |
| **Early Stopping** | patience=3, monitor=val_loss |
| **Checkpoint Strategy** | save_top_k=1, monitor=val_loss, mode=min |
| **Video Sampling** | N=10 frames via np.linspace (uniform) |
| **Video Aggregation** | Mean probability across sampled frames |
| **Model File** | best_model-v3.pt (15.6 MB, state_dict format) |
| **Web Framework** | Gradio (gr.Blocks) |
| **Supported Inputs** | .jpg, .jpeg, .png, .mp4, .mov |
| **Output Classes** | 0 = Real, 1 = Fake |

### Pipeline Summary

```mermaid
graph LR
    A["📁 Data Prep<br/>tools/*"] --> B["🏋️ Training<br/>main_trainer.py"]
    B --> C["💾 Model Export<br/>export_to_pt.py"]
    C --> D["🔍 Inference<br/>web-app.py<br/>classify.py<br/>realeval.py"]
    D --> E["📊 Results<br/>Real / Deepfake<br/>+ Confidence %"]

    style A fill:#6366f1,stroke:#4f46e5,color:#fff
    style B fill:#a855f7,stroke:#9333ea,color:#fff
    style C fill:#f59e0b,stroke:#d97706,color:#fff
    style D fill:#22c55e,stroke:#16a34a,color:#fff
    style E fill:#3b82f6,stroke:#2563eb,color:#fff
```
