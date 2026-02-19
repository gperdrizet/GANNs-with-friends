GANNs with friends documentation
===================================

Welcome to the documentation for **GANNs with friends** - an educational distributed deep learning system where students become part of a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces.

What is a generative adversarial network?
------------------------------------------

A **GAN** is a type of neural network that learns to create realistic images by using two competing models:

- **Generator**: Creates fake images from random noise
- **Discriminator**: Tries to distinguish real images from fake ones

The generator and discriminator play a game: the generator tries to fool the discriminator, while the discriminator gets better at detecting fakes. Through this competition, the generator learns to create increasingly realistic images.

In this project, students train a DCGAN (Deep Convolutional GAN) to generate celebrity faces using the CelebA dataset.

::

    ┌─────────────────────────────────────────────────────────────────┐
    │                    GAN: ADVERSARIAL TRAINING                    │
    └─────────────────────────────────────────────────────────────────┘
    
              ┌──────────────┐              ┌──────────────┐
              │   Random     │              │  Real Images │
              │   Noise      │              │  (CelebA)    │
              │   z ∼ N(0,1) │              │              │
              └──────┬───────┘              └──────┬───────┘
                     │                             │
                     ▼                             ▼
              ┌──────────────┐              ┌──────────────┐
              │  Generator   │              │Discriminator │
              │  G(z)        │──fake img──> │  D(x)        │
              │              │              │              │
              └──────┬───────┘              └──────┬───────┘
                     │                             │
                     │      ┌─────────────┐        │
                     │      │  Backprop   │        │
                     └──────│  Gradients  │────────┘
                            └─────────────┘
                                   │
                            "fool D"   vs   "detect fakes"
                                   │
                            [Nash Equilibrium]

What is distributed training?
------------------------------

Instead of training on a single machine, **distributed training** spreads the work across multiple computers working in parallel. This project uses a unique approach:

- **Students' laptops/GPUs** become workers in a compute cluster
- A **PostgreSQL database** coordinates all communication (no complex networking required!)
- Each worker processes different batches of images and computes gradients
- The coordinator aggregates gradients and updates the model

This "database as coordinator" design makes distributed training accessible in classroom settings without requiring port forwarding or VPN configuration.

::

    ┌─────────────────────────────────────────────────────────────────┐
    │           DISTRIBUTED TRAINING: STUDENT COMPUTE CLUSTER         │
    └─────────────────────────────────────────────────────────────────┘
    
         [Student A]        [Student B]        [Student C]
         Laptop/GPU         Desktop/GPU        Colab GPU
              │                  │                  │
              │ claim_work()     │ claim_work()     │ claim_work()
              ▼                  ▼                  ▼
         ┌─────────┐        ┌─────────┐        ┌─────────┐
         │ batch   │        │ batch   │        │ batch   │
         │ 0-31    │        │ 32-63   │        │ 64-95   │
         └────┬────┘        └────┬────┘        └────┬────┘
              │                  │                  │
         compute_∇()         compute_∇()        compute_∇()
              │                  │                  │
              │ upload_grads()   │                  │
              └────────┬─────────┴──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   PostgreSQL   │  ← coordination hub
              │   Database     │     (no networking!)
              └────────┬───────┘
                       │
              aggregate_gradients()
                       │
                       ▼
              ┌────────────────┐
              │  Coordinator   │  ← instructor runs this
              │  (Instructor)  │
              └────────┬───────┘
                       │
              update_model_weights()
                       │
                       ▼
              [Iteration N+1: create new work units...]

Why this project?
-----------------

**GANNs with friends** is designed to be:

**Educational**
    Students learn about GANs, distributed systems, gradient aggregation, and fault tolerance through hands-on experience.

**Accessible**
    Works on CPU or GPU, runs on Google Colab, and requires minimal setup. The database-centric architecture works across firewalls.

**Practical**
    Demonstrates real distributed training concepts used in industry while keeping the codebase simple and readable.

**Fun**
    Watch the GAN evolve in real-time as the class collaborates to generate realistic faces. See how adding more workers speeds up training.

A note on production distributed training
------------------------------------------

This project is designed for **education**, not production workloads. Real high-performance distributed training uses dedicated infrastructure:

- **Compute clusters** with high-speed interconnects (InfiniBand, NVLink)
- **Optimized frameworks** like Horovod, DeepSpeed, or PyTorch Distributed Data Parallel
- **Direct GPU-to-GPU communication** for fast gradient synchronization
- **Specialized hardware** like multi-GPU nodes, TPU pods, or cloud clusters

Production systems prioritize performance and can achieve near-linear scaling with hundreds of GPUs. This educational project prioritizes accessibility and learning - it uses a database for coordination, works across heterogeneous hardware, and can run on laptops with integrated graphics.

The concepts you learn here (gradient aggregation, fault tolerance, work distribution) apply to production systems, but the implementation is simplified for teaching.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting-started/overview
   getting-started/installation
   getting-started/quick-start

.. toctree::
   :maxdepth: 1
   :caption: Setup paths

   setup/google-colab
   setup/dev-container
   setup/native-python
   setup/conda
   setup/local-training

.. toctree::
   :maxdepth: 1
   :caption: User guides

   guides/students
   guides/instructors
   guides/configuration
   guides/monitoring

.. toctree::
   :maxdepth: 1
   :caption: Architecture

   architecture/overview
   architecture/database
   architecture/coordinator
   architecture/worker
   architecture/models

.. toctree::
   :maxdepth: 1
   :caption: Development approach

   development-approach

.. toctree::
   :maxdepth: 1
   :caption: Features

   features/distributed-training
   features/huggingface-integration
   features/cpu-support

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/models
   api/database
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional resources

   resources/troubleshooting
   resources/performance
   resources/contributing
   resources/faq

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
