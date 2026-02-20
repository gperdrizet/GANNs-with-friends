# Background

This page provides deeper technical background on GANs and distributed training for students who want to understand the underlying concepts.

## How GANs work

Generative Adversarial Networks (GANs) are a class of deep learning models that learn to generate new data by pitting two neural networks against each other in a game-theoretic framework.

### The adversarial game

A GAN consists of two competing networks:

```
Random Noise ──> [Generator] ──> Generated Image ──┐
                                                    ├──> [Discriminator] ──> Real or Fake?
                     Real Image (CelebA) ──────────┘
```

**Generator (G)**: Takes random noise as input and produces synthetic images. Its goal is to create images realistic enough to fool the discriminator.

**Discriminator (D)**: Receives both real images (from the training dataset) and fake images (from the generator). Its goal is to correctly classify which images are real and which are fake.

### The training objective

GANs are trained using a minimax game where:
- The **discriminator** tries to maximize its ability to distinguish real from fake
- The **generator** tries to minimize the discriminator's success

We use **Binary Cross-Entropy (BCE) loss** to train both networks. BCE is the standard loss function for binary classification problems - it measures how well a model's predicted probabilities match the true labels (0 or 1).

**Training the Discriminator:**
The discriminator is trained to output values close to 1 for real images and close to 0 for fake images. We compute BCE loss twice per step:
1. Feed real images with label=1, penalizing when D outputs low values
2. Feed generated (fake) images with label=0, penalizing when D outputs high values

The discriminator's gradients push it to better distinguish real from fake.

**Training the Generator:**
The generator is trained to fool the discriminator. We feed generated images through the discriminator, but use label=1 (pretending they're real). The BCE loss penalizes the generator when the discriminator correctly identifies fakes as fake.

The key insight: gradients flow backward through the frozen discriminator into the generator, teaching it what features make images look "more real" to the discriminator.

### Training dynamics

Each training step involves:

1. **Train Discriminator**:
   - Show it real images (label = 1) and compute loss
   - Show it generated images (label = 0) and compute loss
   - Backpropagate and update discriminator weights

2. **Train Generator**:
   - Generate fake images
   - Ask discriminator to classify them (but freeze D's weights)
   - Backpropagate through D into G, update generator weights

```
┌─────────────────────────────────────────────────────────┐
│                    GAN Training Loop                    │
├─────────────────────────────────────────────────────────┤
│  Step 1: Train Discriminator                            │
│  ┌─────────┐      ┌─────────┐                           │
│  │  Real   │──────│ D(real) │──> Loss: -log(D(real))    │
│  │ Images  │      └─────────┘                           │
│  └─────────┘                                            │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐          │
│  │  Noise  │──────│    G    │──────│ D(fake) │──> Loss  │
│  └─────────┘      └─────────┘      └─────────┘          │
│                                     -log(1-D(fake))     │
│                                                         │
│  Backprop: ∂L/∂θ_D computed, D weights updated          │
├─────────────────────────────────────────────────────────┤
│  Step 2: Train Generator                                │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐          │
│  │  Noise  │──────│    G    │──────│    D    │──> Loss  │
│  └─────────┘      └─────────┘      └─────────┘          │
│                                     -log(D(G(z)))       │
│                                                         │
│  Backprop: gradients flow through frozen D into G       │
│  Only G weights updated (D frozen)                      │
└─────────────────────────────────────────────────────────┘
```

### GAN applications beyond image generation

While this project focuses on generating faces, GANs have many other applications:

**Computer Vision**
- **Super-resolution**: Enhance low-resolution images (SRGAN, ESRGAN)
- **Image inpainting**: Fill in missing or damaged regions
- **Style transfer**: Apply artistic styles to photos (CycleGAN)
- **Image-to-image translation**: Convert sketches to photos, day to night, etc. (pix2pix)

**Audio & Music**
- **Voice synthesis**: Generate realistic speech (WaveGAN)
- **Music generation**: Create novel musical compositions
- **Voice conversion**: Transform one voice to sound like another

**Science & Medicine**
- **Drug discovery**: Generate novel molecular structures
- **Medical imaging**: Synthesize training data, augment datasets
- **Protein structure**: Generate plausible protein conformations

**Other Domains**
- **Text generation**: Though transformers dominate, GANs have been explored (SeqGAN)
- **Video synthesis**: Generate realistic video sequences
- **3D object generation**: Create 3D models from 2D images
- **Data augmentation**: Generate synthetic training data to improve classifiers
- **Anomaly detection**: Discriminator learns normal data distribution

### Why GANs are challenging

GANs can be difficult to train due to several factors:

- **Mode collapse**: Generator produces limited variety of outputs
- **Training instability**: Loss oscillates instead of converging
- **Vanishing gradients**: Discriminator becomes too good, giving generator no useful signal
- **Hyperparameter sensitivity**: Learning rate, architecture, and batch size all matter significantly

The DCGAN architecture we use incorporates best practices that help stabilize training:
- Batch normalization in generator and discriminator
- LeakyReLU activations in discriminator
- Strided convolutions instead of pooling
- Adam optimizer with specific β parameters

## Distributed training for machine learning

Training deep learning models requires enormous computation. Distributing this work across multiple machines dramatically reduces training time.

### Why distribute training?

- **Faster iteration**: Train models in hours instead of days
- **Larger models**: Fit models that don't fit on a single GPU
- **Better utilization**: Use idle compute resources efficiently
- **Collaboration**: Multiple participants contribute to a shared goal

### Types of distributed training

#### Data parallelism (what we use)

Each worker processes different data with the same model:

```
┌─────────────────────────────────────────────────────────┐
│                   Data Parallelism                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Training Data: [Batch 1] [Batch 2] [Batch 3] [Batch 4] │
│                     │         │         │         │     │
│                     ▼         ▼         ▼         ▼     │
│                 ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ │
│                 │Worker1│ │Worker2│ │Worker3│ │Worker4│ │
│                 │(GPU 1)│ │(GPU 2)│ │(GPU 3)│ │(GPU 4)│ │
│                 └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ │
│                     │         │         │         │     │
│                     ▼         ▼         ▼         ▼     │
│                 [Grad 1]  [Grad 2]  [Grad 3]  [Grad 4]  │
│                     │         │         │         │     │
│                     └────────┬┴─────────┴─────────┘     │
│                              ▼                          │
│                    ┌─────────────────┐                  │
│                    │ Average/Reduce  │                  │
│                    └────────┬────────┘                  │
│                             ▼                           │
│                    [Aggregated Gradient]                │
│                             │                           │
│                             ▼                           │
│                    ┌─────────────────┐                  │
│                    │ Update Weights  │                  │
│                    └─────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**Advantages**:
- Simple to implement
- Scales well with more workers
- Each worker sees different data, improving gradient estimates

#### Model parallelism

Different workers hold different parts of the model:

```
Input ──> [Worker 1: Layers 1-4] ──> [Worker 2: Layers 5-8] ──> Output
```

Used when models are too large for a single GPU (e.g., large language models with billions of parameters).

**Challenges with model parallelism**:
- **Pipeline bubbles**: Workers sit idle waiting for activations from previous stages. With 4 pipeline stages, up to 75% of compute can be wasted during the filling/draining phases.
- **Communication overhead**: Activations between layers must be sent between workers on every forward pass, and gradients on every backward pass.
- **Memory imbalance**: Different layers have different sizes; balancing memory across workers is non-trivial.
- **Complex implementation**: Requires careful partitioning of the model and coordination of forward/backward passes.
- **Debugging difficulty**: Errors can be hard to trace across distributed model segments.

Modern large model training (GPT-4, LLaMA, etc.) combines multiple strategies: tensor parallelism within nodes, pipeline parallelism across nodes, and data parallelism across groups of nodes.

### Gradient aggregation strategies

#### Synchronous (what we use)

All workers must complete before updating:

```
Time ──────────────────────────────────────────→

Worker 1: ████████░░░░░░░░░░████████░░░░░░░░░░
Worker 2: ██████████████░░░░████████████░░░░░░
Worker 3: ████████████░░░░░░██████████████░░░░
                      ↑                   ↑
               Sync barrier         Sync barrier
               (aggregate)          (aggregate)
```

##### Advantages: Consistent, predictable convergence
##### Disadvantages: Slowest worker determines pace

#### Asynchronous (alternative approach)

Workers update independently without waiting:

```
Worker 1: ████↑████↑████↑████↑
Worker 2: ██████↑██████↑██████↑
Worker 3: ████████↑████████↑████
          ↑ = update weights
```

##### Advantages: No idle time, faster wall-clock
##### Disadvantages: Stale gradients, less stable convergence

### Our approach: Database-coordinated synchronous training

This project uses a hybrid approach:

1. **Partial synchronization**: Wait for N workers (not all) before updating
2. **Database as message queue**: No direct worker-to-worker communication
3. **Weighted averaging**: Workers contribute proportionally to samples processed

```
┌─────────────────────────────────────────────────────────┐
│           Database-Coordinated Training                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │
│  │Worker 1│  │Worker 2│  │Worker 3│  │Worker 4│         │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘         │
│      │           │           │           │              │
│      ▼           ▼           ▼           ▼              │
│  ┌──────────────────────────────────────────────┐       │
│  │              PostgreSQL Database             │       │
│  │  ┌────────┐  ┌──────────┐  ┌────────────┐    │       │
│  │  │  Work  │  │ Gradients│  │   Model    │    │       │
│  │  │ Units  │  │          │  │  Weights   │    │       │
│  │  └────────┘  └──────────┘  └────────────┘    │       │
│  └──────────────────────────────────────────────┘       │
│                          ▲                              │
│                          │                              │
│                   ┌──────┴──────┐                       │
│                   │ Coordinator │                       │
│                   │  (Aggregates│                       │
│                   │   gradients)│                       │
│                   └─────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

This design is ideal for educational settings:
- **No complex networking**: Students don't need port forwarding or VPNs
- **Fault tolerant**: Workers can join/leave without disruption
- **Transparent**: All state visible in database for debugging

## Next steps

- [Installation](installation.md) - Set up your environment
- [Quick start](quick-start.md) - Get running in 5 minutes
