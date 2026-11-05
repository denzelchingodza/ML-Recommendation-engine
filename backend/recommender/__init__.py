"""Startup recommender for investors.

Given an investor, suggest the startups they are most likely to back,
based on what they have backed before.

Layout (each part is added in its own phase):
    data/    load and split the Crunchbase 2013 data
    maths/   all the maths, one job per file, each tested against a trusted library
    models/  uses the maths to make recommendations
    api/     serves recommendations to the frontend
"""

__version__ = "0.0.1"
