# Distributed GAN training with students as workers

An educational distributed deep learning system where students become part of a compute cluster to train a GAN (Generative Adversarial Network) to generate images.

## Concept

This project demonstrates distributed machine learning by:
- Using students' computers as a distributed compute cluster
- Coordinating training through a PostgreSQL database (no complex networking!)
- Training a DCGAN to generate realistic images
- Teaching distributed systems, parallel training, and GANs simultaneously

## Architecture

**Main process (instructor/admin):**
- Creates work units (batches of image indices)
- Aggregates gradients from workers
- Applies optimizer steps
- Tracks training progress

**Worker process (students/workers):**
- Polls database for available work
- Computes gradients on assigned image batches
- Uploads gradients back to database
- Runs continuously until training completes

**PostgreSQL database:**
- Stores model weights, gradients, work units
- Acts as communication hub (no port forwarding needed!)
- Tracks worker statistics for monitoring
- **Note**: instructor/admin needs to set-up student-accessible SQL database

## Documentation

**[Full documentation](https://gperdrizet.github.io/GANNs-with-friends)**

Quick links:
- [Getting Started](https://gperdrizet.github.io/GANNs-with-friends/getting-started/overview.html) - Introduction and concepts
- [Installation Guide](https://gperdrizet.github.io/GANNs-with-friends/getting-started/installation.html) - Choose your setup path
- [Student Guide](https://gperdrizet.github.io/GANNs-with-friends/guides/students.html) - How to participate as a worker
- [Instructor Guide](https://gperdrizet.github.io/GANNs-with-friends/guides/instructors.html) - Running the coordinator
- [Configuration Reference](https://gperdrizet.github.io/GANNs-with-friends/guides/configuration.html) - All config options
- [Architecture](https://gperdrizet.github.io/GANNs-with-friends/architecture/overview.html) - System design details
- [FAQ](https://gperdrizet.github.io/GANNs-with-friends/resources/faq.html) - Frequently asked questions

## Quick start

Choose your installation path:

| Setup Path | Best For | GPU Required | Documentation |
|------------|----------|--------------|---------------|
| **Dev Container** † | Full development environment | Optional | [Setup guide](https://gperdrizet.github.io/GANNs-with-friends/setup/dev-container.html) |
| **Native Python** | Direct local control | Optional | [Setup guide](https://gperdrizet.github.io/GANNs-with-friends/setup/native-python.html) |
| **Conda** | Conda users | Optional | [Setup guide](https://gperdrizet.github.io/GANNs-with-friends/setup/conda.html) |
| **Google Colab** | Zero installation, free GPU | No (provided) | [Setup guide](https://gperdrizet.github.io/GANNs-with-friends/setup/google-colab.html) |
| **Local Training** | Single GPU, no database | Optional | [Setup guide](https://gperdrizet.github.io/GANNs-with-friends/setup/local-training.html) |

† Recommended configuration

- **New to the project?** Start with the [Getting Started Guide](https://gperdrizet.github.io/GANNs-with-friends/getting-started/overview.html).
- **For students:** See the [Student Guide](https://gperdrizet.github.io/GANNs-with-friends/guides/students.html) for how to participate as a worker.
- **For instructors:** See the [Instructor Guide](https://gperdrizet.github.io/GANNs-with-friends/guides/instructors.html) for running the coordinator and managing training.

## Features

- **Database-coordinated training**: No complex networking, works across firewalls
- **Fault tolerant**: Workers can disconnect/reconnect, automatic work reassignment
- **Flexible hardware**: CPU and GPU workers can participate together
- **Educational**: Learn distributed systems, GANs, and parallel training

## What students learn

- **Distributed systems**: Coordination, fault tolerance, atomic operations
- **Deep learning**: GAN training, gradient aggregation, data parallelism
- **Practical skills**: PostgreSQL, PyTorch, collaborative computing

## Contributing

This is an educational project! Contributions welcome:
- Bug fixes and improvements
- Additional GAN architectures
- Gradient compression techniques

See the [Contributing Guide](https://gperdrizet.github.io/GANNs-with-friends/resources/contributing.html) for more details.

## License

MIT License - See LICENSE file for details