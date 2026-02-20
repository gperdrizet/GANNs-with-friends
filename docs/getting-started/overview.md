# Overview

**GANNs with friends** is an educational distributed deep learning system designed to teach students about:

- Distributed machine learning systems
- Generative Adversarial Networks (GANs)
- Database-coordinated computing
- Collaborative training at scale

## What makes this project unique?

### Database-coordinated distributed training

Unlike traditional distributed systems that require complex networking setup, this project uses PostgreSQL as a coordination layer. This means:

- No port forwarding or VPN setup required
- Students can participate from anywhere with internet access
- Simple to set up and manage
- Fault-tolerant by design

### Multiple participation paths

We support four different ways to participate, removing barriers to entry:

1. **Google Colab** - Zero installation, free GPU
2. **Dev container** - Full development environment
3. **Native Python** - Direct local installation
4. **Conda** - For conda users

### Educational focus

This project is designed from the ground up for teaching:

- Clear separation between coordinator and worker roles
- Well-documented codebase with extensive comments
- Multiple difficulty levels (simple worker to full coordinator)
- Real-time visualization of training progress

## Key concepts

### Distributed data parallel training

Each worker:
- Receives a unique batch of images
- Computes gradients independently  
- Uploads results to the shared database
- Continues with the next available work unit

The coordinator:
- Creates work units (batches of image indices)
- Waits for N workers to complete their work
- Aggregates the gradients
- Updates the model weights
- Publishes new weights for the next iteration

### GAN architecture

The project trains a DCGAN (Deep Convolutional GAN) to generate celebrity faces:

- **Generator**: Transforms 100D random noise into 64x64 RGB images
- **Discriminator**: Learns to distinguish real from generated images
- **Adversarial training**: Generator and discriminator compete, driving improvement

### Database as message queue

PostgreSQL provides:
- Atomic work unit claiming with `FOR UPDATE SKIP LOCKED`
- BLOB storage for model weights and gradients
- Worker heartbeat tracking
- Training state persistence

## Project outcomes

After completing this project, students understand:

- How modern distributed training systems work
- The challenges of coordinating multiple workers
- GAN architecture and training dynamics
- Database transactions and concurrency
- Practical considerations in distributed ML

## Next steps

- [Background](background.md) - Deeper technical background on GANs and distributed training
- [Installation](installation.md) - Set up your environment
- [Quick start](quick-start.md) - Get running in 5 minutes
