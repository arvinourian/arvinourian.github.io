---
layout: post
title: "From Dataset to ML Application in Minutes: A Practical Guide to Hugging Face Spaces"
date: 2025-11-23
---

You've built a machine learning model. It works great in your Jupyter notebook. Now what? This is the question that haunts data science teams everywhere. The journey from a working prototype to a deployed, user-facing application is often measured in weeks or months. This is not because machine learning is hard, but because deployment is tedious and complicated.

In this post we explore Hugging Face Spaces, a platform that promises to streamline this process while also providing a more user-friendly environment to demo your ML project. I tested it by building and deploying a movie recommendation system using the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/).

You can view and test the completed [demo space here](https://huggingface.co/spaces/Anourian/movie-recommender) , aswell as all the [corresponding files](https://huggingface.co/spaces/Anourian/movie-recommender/tree/main)

## The Deployment Problem in ML Production

Before diving into the solution, let's understand the problem. In a typical ML workflow, getting a model into production requires:

- **Infrastructure setup:** Provisioning servers, configuring networking, setting up load balancers
- **Containerization:** Writing Dockerfiles, managing dependencies, handling environment variables
- **Web framework integration:** Building APIs with Flask/FastAPI, creating frontend interfaces
- **DevOps configuration:** CI/CD pipelines, monitoring, logging, certificates

For a data scientist who just wants stakeholders to try their model, this overhead is massive. Even with cloud platforms like AWS or GCP, you're looking at hours of configuration for a simple demo.

# What is Hugging Face Spaces?

Hugging Face Spaces is a hosting platform specifically designed for ML applications. It supports multiple frameworks including Gradio (for quick ML interfaces), Streamlit (for data apps), and static HTML/Docker for custom deployments.

The core value proposition of this is to push your code to a Git repository, and Spaces handles everything else. This includes container building, dependency installation, hosting, and providing a public URL.

# Building a Movie Recommender: A Practical Walkthrough

To test Hugging Face Spaces in a realistic scenario, I use the MovieLens 1M dataset to build a simple movie recommendation system. This program allows you to select a genre to search and provides recommendations based on the highest ratings found in that genre. There are also sliders that allow you to select a minimum number of ratings and number of results. Lastly, there is a search function to easily explore known movie titles, and their associated genres and ratings.

Hugging Face Spaces provides value here by allowing less DevOps oriented stakeholders to demo the dataset and demo the recommendation system. As a data engineer you can also get value out of the easy to navigate UI that this platform provides to explore your dataset.

## The Dataset: [MovieLens 1M]((https://grouplens.org/datasets/movielens/))

I used the MovieLens 1M dataset, which contains 1 million ratings from 6,040 users on 3,883 movies. This is a standard benchmark dataset that mirrors the kind of data a real streaming service would have.

## Step 1: Create the Application

I chose [Gradio](https://www.gradio.app/docs) as the framework because it's designed specifically for ML interfaces and requires minimal frontend knowledge. Here's the core logic:

```python
# app.py - Core recommendation function
def recommend_movies(genre, min_ratings, num_results):
    filtered = movies[movies['Genres'].str.contains(genre)]
    filtered = filtered[filtered['num_ratings'] >= min_ratings]
    return filtered.sort_values('avg_rating', ascending=False)
```

Gradio handles the UI automatically. With about 80 lines of Python, I will have a complete application with dropdowns, sliders, and a results table.

## Step 2: Prepare for Deployment

Deploying to Spaces requires just three files:

1. **app.py**: Your Gradio application
2. **requirements.txt**: Python dependencies (gradio, pandas)
3. **Data files:** movies.dat, ratings.dat, users.dat

## Step 3: Deploy to Hugging Face Spaces

The deployment process is straightforward:

After creating a Hugging Face Account, navigate to [https://huggingface.co/spaces](https://huggingface.co/spaces) and click create a "New Space"

![Creating a new Space](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image4.png?raw=true)

Provide a Space name, short description, and choose the SDK for your project. In this case we are using Gradio.

![Space configuration](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image7.png?raw=true)

You also have the option of selecting between templates and hardware, public/private, and Space Dev Mode which allows you to work on your Space remotely using SSH or VS Code.

One completed, click "Create Space"

![Create Space button](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image3.png?raw=true)

One the Space is created you should be directed to the URL `https://huggingface.co/spaces/[username]/[space-name]`, and there should be instructions on how to get started. Here we use Hugging Face Space's own file UI to upload the needed files.

![Space instructions](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image8.png?raw=true)

Click "Files" > "Contribute" > "Upload files"

![Upload files menu](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image2.png?raw=true)

Then its as simple as dragging and dropping our demo's files and clicking "Commit changes to main"

![Uploading files](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image5.png?raw=true)

After a few minutes, we navigate back to `https://huggingface.co/spaces/[username]/[space-name]` and see that the application is being deployed without us having to mess with Docker, AWS or any other tedious tools when we just want a simple demo to run.

![Application building](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image6.png?raw=true)

![Deployed application](https://github.com/arvinourian/arvinourian.github.io/blob/main/images/image1.png?raw=true)

# Analysis: Strengths and Limitations

## Strengths

• **Speed of deployment:** This is the main advantage of Hugging Face Spaces, allowing you to go from notebook to shareable URL in minutes. This changes how teams communicate and iterate demo projects, and project managers can see features the same day they're requested.

• **Low barrier to entry:** Data scientists without DevOps experience can deploy applications. This democratizes who can ship software..

• **Community and discoverability:** Spaces are public by default, making it easy to share work and learn from others. The Hugging Face community has thousands of example applications.

## Limitations

Most limitations are relevant to the Free-tier that we are using for this demonstration of the application, but if we are weighing HFS with our own inbuilt deployment schema they are relevant.

• **Cold starts:** Free-tier Spaces like the one we are using sleep after inactivity. The first request can take 30-60 seconds while the container spins up. This isn't suitable for production APIs requiring consistent latency.

• **Resource constraints:** The free tier (2 vCPU, 16GB RAM, 50gb of non-persistent disk space) is fine for demos but may struggle with large models or high traffic. Upgrading to persistent or GPU instances costs more.

## When to Use (and Not Use) Hugging Face Spaces

**Great for:** Prototypes, demos, hackathons, internal tools, proof-of-concepts, educational projects, portfolio pieces

**Not ideal for:** Production APIs with SLAs, applications requiring low latency, sensitive/proprietary data, complex microservice architectures

# Conclusion: The Right Tool for Rapid Iteration

Hugging Face Spaces excels at getting ML applications in front of users fast. For our movie recommendation scenario, it transformed a genre recommendation algorithm into a shareable demo in under 10 minutes. This is something that could traditionally take a day or more with conventional deployment approaches.

It's not a replacement for production infrastructure, but it doesn't try to be. Instead, it fills a critical gap in the ML workflow: the space between "it works in my notebook" and "let's invest in proper infrastructure". For teams that want to validate ideas quickly, gather stakeholder feedback, or share work publicly, Hugging Face Spaces is a powerful addition to the MLOps toolkit.
