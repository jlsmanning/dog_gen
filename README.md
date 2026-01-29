# Dog Breed Classifier

A genetics-informed dog breed classification system implementing research on **error characterization in AI classification**. This project addresses the question: *"How wrong is wrong?"*  and how to evaluate not just whether models are accurate, but how severe their errors are.

## Live Demo
Try the deployed model: https://jlsmanning-dog-gen-hf.hf.space/docs

## Research Background

This work was presented at the AI4SE & SE4AI Workshop 2022:

> J. Manning, R. Pless, and Z. Szajnfarber, "How wrong is wrong? Richer characterization of AI classification error for real world applications," Presented at AI4SE & SE4AI Workshop 2022, Hoboken, NJ, USA, Sept. 2022. Available: https://sercuarc.org/wp-content/uploads/2025/06/1664289267.D1.T2.S4_manning_how_wrong_is_wrong.pdf

### Motivation

Traditional ML evaluation focuses on accuracy, but for safety-critical applications (e.g., friend-or-foe identification in military systems), **the nature of errors matters**. Confusing similar fighter jets might be less severe than confusing a fighter jet with a passenger plane. **How a model fails can matter as much as how often it fails.**

This project uses dog breed classification as a case study to demonstrate error characterization using genetic distance between breeds. Why dogs? Dogs come in many shapes and sizes even though they are all still the same species. Some breeds are easy to tell apart, while others can be very difficult to discriminate. There is also a ground truth available in the form of research into dog genetic relationships.

## Key Contributions

1. **Error Magnitude Quantification**: Uses genetic distance matrix to measure "how bad" misclassifications are, where "bad" is a large genetic difference.
2. **Comparative Analysis**: Shows ResNet50 not only achieves higher accuracy (89.4% vs 76.1%) but also makes smaller errors (mean distance 0.390 vs 0.431, P=0.0002)
3. **Genetics-Informed Loss**: Experimental loss function that incorporates genetic distances as soft labels to encourage "less wrong" predictions

## Research Findings

### Quantitative Results
- **ResNet18**: 76.12% accuracy, mean error distance 0.431
- **ResNet50**: 89.40% accuracy, mean error distance 0.390 (significantly better, P=0.0002)
- **ResNet18 + dist_loss**: Mean error distance 0.3057 (improved), but 3% accuracy drop

### Key Insight
Deeper networks don't just classify better; they can also make more "forgivable" mistakes when they fail.

## Real-World Applications

While this project uses dog breeds as a case study, the methodology generalizes to any classification problem where errors have varying severity:

### Military & Defense
- **Aircraft identification**: Confusing two allied fighter types vs. friendly/hostile misidentification
- **Vehicle recognition**: Confusing civilian vs. military vehicles
- **Threat assessment**: Distinguishing weapon types with hierarchical risk levels

### Medical Imaging
- **Cancer staging**: Confusing Stage 2A vs 2B is less severe than Stage 1 vs Stage 4
- **Disease diagnosis**: Within-category errors (viral vs. bacterial pneumonia) vs. across-category errors (pneumonia vs. lung cancer)

### Autonomous Vehicles
- **Object detection**: Confusing sedan vs. SUV is less critical than vehicle vs. pedestrian
- **Traffic sign recognition**: Speed limit variations vs. stop sign misclassification

### Financial Services
- **Fraud detection**: False positives on legitimate transactions vs. missing actual fraud
- **Risk assessment**: Misclassifying risk levels within the same category vs. across categories

### How to implement with a different problem

The main idea here, measuring how bad mistakes are using domain knowledge, applies directly to safety-critical work where some errors matter much more than others.

1. **Define the hierarchy**: Identify the structure that defines "distance" between classes. 

2. **Quantify distances**: Create a distance matrix between classes,

3. **Evaluate error quality**: Use the distance metric to assess not just accuracy but error severity,

4. **Optional: Train with hierarchy awareness**: Incorporate distance information into loss functions. (As my dog breed experiment showed, this can be a trade-off with accuracy)


## Project Structure

```
dog-breed-classifier/
├── config/                      # Configuration files
│   ├── train_config.yaml
│   └── inference_config.yaml
├── data/                        # Data loading and preprocessing
│   ├── breed_mapping.py
│   ├── datasets.py
│   ├── transforms.py
│   └── genetic_distance.py
├── models/                      # Model definitions
│   ├── resnet_classifier.py
│   └── model_loader.py
├── training/                    # Training pipeline
│   ├── trainer.py
│   ├── train.py
│   └── evaluate.py
├── inference/                   # Inference and API
│   ├── predictor.py
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
├── utils/                       # Utilities
│   ├── visualization.py
│   └── metrics.py
├── experiments/                 # Analysis scripts
│   ├── embedding_analysis.py
│   ├── neighbor_search.py
│   └── statistical_tests.py
├── saved_models/               # Trained model weights
├── outputs/                    # Training outputs and analysis results
└── tests/                      # Unit tests
```

## Dataset

- **Images**: Stanford Dogs Dataset (20,580 images, 120 breeds) - http://vision.stanford.edu/aditya86/ImageNetDogs/
- **Genetic Data**: From Parker et al. (2017), *Cell Reports* - genetic distance matrix between dog breeds

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/jlsmanning/dog_gen.git
cd dog_gen
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install package in development mode:
```bash
pip install -e .
```

### Dataset Setup

1. **Training Dataset**: Organize your dog breed images as:
   ```
   images/
   ├── train/
   │   ├── n02085620-Chihuahua/
   │   ├── n02088364-beagle/
   │   └── ...
   ├── val/
   │   └── (same structure)
   └── test/
       └── (same structure)
   ```
   

2. **Genetic Data**: Already included in `data/genetic_data/`:
   - `Breeds.txt` - Breed assignments for each sample
   - `distances_all.txt` - Pairwise genetic distances

### Configuration

1. Copy the example config:
   ```bash
   cp config/train_config.example.yaml config/train_config.yaml
   ```

2. If your data is not in the same directory, either edit `config/train_config.yaml` and update
   ```yaml
   paths:
     dataset: '/path/to/your/images'
   ```
   or make a symlink called "images" in the main directory and link it to the dataset
3. Run training:
   ```bash
   python training/train.py
   ```

### Docker Setup

```bash
cd inference/docker
docker-compose up --build
```

## Usage

### Training

1. Configure training in `config/train_config.yaml`

2. Run training:
```bash
python training/train.py
```

Or with custom config:
```bash
python training/train.py config/custom_config.yaml
```

### Evaluation

```bash
python training/evaluate.py
```

Or evaluate a specific model:
```bash
python training/evaluate.py config/train_config.yaml saved_models/resnet50_score_loss_model.pth
```

### Experiments

Extract embeddings:
```bash
python experiments/embedding_analysis.py
```

Visualize nearest neighbors:
```bash
python experiments/neighbor_search.py
```

Statistical comparisons:
```bash
python experiments/statistical_tests.py
```

### API Inference

Start the API server:
```bash
uvicorn inference.api.app:app --host 0.0.0.0 --port 8000
```

Access interactive documentation at: `http://localhost:8000/docs`

Example API usage:
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("dog_image.jpg", "rb")}
params = {"top_k": 5}

response = requests.post(url, files=files, params=params)
print(response.json())
```

### Python API

```python
from inference.predictor import load_predictor

# Load predictor
predictor = load_predictor(
    model_path='saved_models/resnet18_score_loss_model.pth',
    config_path='config/inference_config.yaml'
)

# Make prediction
result = predictor.predict('path/to/dog/image.jpg', top_k=5)
print(f"Predicted breed: {result['top_prediction']['breed_name']}")
print(f"Confidence: {result['top_prediction']['confidence']:.2f}%")
```

## Model Architectures

This project supports:
- **ResNet18**: Faster training, good for experimentation
- **ResNet50**: Higher accuracy, more parameters

## Loss Functions

### Standard Cross-Entropy (score_loss)
Traditional classification loss treating all misclassifications equally.

### Genetics-Informed Loss (dist_loss)
Novel loss function that converts genetic distances into soft labels:
- Similar breeds (small genetic distance) receive higher probability mass
- Dissimilar breeds receive lower probability mass
- Encourages the model to make "less wrong" mistakes
- Controlled by `label_threshold` parameter (default: 0.3)

**Trade-off**: Initial experiments show dist_loss reduces mean error distance but may slightly decrease overall accuracy. This represents a tunable parameter space for specific application requirements.

## Configuration

Key configuration options in `config/train_config.yaml`:

```yaml
model:
  architecture: 'resnet18'  # or resnet50
  pretrained: true

training:
  num_epochs: 20
  loss_type: 'score_loss'  # or dist_loss
  label_threshold: 0.3     # for dist_loss

  optimizer:
    lr: 0.001
    momentum: 0.9

  scheduler:
    step_size: 5
    gamma: 0.2
```

## Evaluation Metrics

The project tracks:
- **Classification accuracy**: Standard accuracy metric
- **Genetic distance of errors**: Mean/median genetic distance for misclassifications
- **Error visualization**: Side-by-side comparisons of misclassified breeds
- **Statistical tests**: T-tests comparing different models
- **Error distribution analysis**: Histograms and density plots of error distances

## API Documentation

Once the server is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

### API Endpoints

- `POST /predict` - Predict breed from uploaded image
- `POST /predict_batch` - Predict breeds from multiple images (max 10)
- `GET /health` - Health check and model status
- `GET /demo` - Run prediction on a random exemplar image
- `GET /exemplars` - List all available breeds for testing
- `GET /predict_exemplar/{breed}` - Predict using a specific breed's exemplar (accepts breed name like "beagle" or "Golden Retriever")
- `GET /` - API information

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{manning2022howwrong,
  author = {Manning, Justine and Pless, Robert and Szajnfarber, Zoe},
  title = {How wrong is wrong? Richer characterization of AI classification error for real world applications},
  booktitle = {AI4SE \& SE4AI Workshop 2022},
  year = {2022},
  month = {September},
  address = {Hoboken, NJ, USA},
  url = {https://sercuarc.org/wp-content/uploads/2025/06/1664289267.D1.T2.S4_manning_how_wrong_is_wrong.pdf}
}
```

## References

Parker, H. G., Dreger, D. L., Rimbault, M., Davis, B. W., Mullen, A. B., Carpintero-Ramirez, G., & Ostrander, E. A. (2017). Genomic Analyses Reveal the Influence of Geographic Origin, Migration, and Hybridization on Modern Dog Breed Development. *Cell Reports*, 19(4), 697-708. https://doi.org/10.1016/j.celrep.2017.03.079

Khosla, A., Jayadevaprakash, N., Yao, B., & Fei-Fei, L. (2011). Novel Dataset for Fine-Grained Image Categorization. *First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition*, Colorado Springs, CO.

## Acknowledgments

- Genetic distance data from Parker et al. (2017)
- Dog breed images from Stanford Dogs Dataset
- Built with PyTorch and FastAPI
- Research conducted at George Washington University

## Contact

- Author: Justine Manning
- Email: jlsmanning@gwu.edu
- GitHub: [@jlsmanning](https://github.com/jlsmanning)
