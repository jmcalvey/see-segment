# Project Proposal

By James McAlvey

## Summary

This project has two parts, first I will explore SEE-Segment and get it running as well as fix any issues that need to be dealt with. The existing library will be used to prepare benchmark tests for the second part. The second part will involve updating the current genetic search algorithm library to something that was developoed more recently and is more robust. The research question for this semester is can I implement a new core library into the software and improve the performance?
## Overview

Image segmentation is the process of dividing an image into meaningful regions or objects. They allow a machine learning model to efficiently analyze the image and learn information about the isolated charachteristics that users are interested in. SEE-Segment is an existing research software project that uses a genetic or evolutionary processing approach to segmentation.
The project goal is to update the software using a more modern tool and making the software more portable and robust.
## Software or Project Description

The software I will be working with is the existing SEE-Segment repository, an existing image segmentation library. The goal will be to switch out the genetic processing library for something more modern while maintaining similar or better performance. First the existing software will be tested and benchmarked and then the new core library will be implemented and tested. The goal will be to improve performance while also making the software easier for researchers to use. 
## Project Goals and Timeline

The short-term goal is getting the current SEE-Segment software running, the medium-term goal is understanding how the genetic search libraries work and how to switch them out, and the long-term goal is to have an improved SEE-Segment with a more powerful library at its core. With about 14 weeks left in the semester there is a tentative deadline every 4-5 weeks to finish one stage and move onto the next. 
## Methods and Workflow






The software design will be implemented using python with github for version control. Software engineering practices will include version control, environment management, documentation, automated testing where practical, and an organized repository on github. Simple tests and debugging strategies will be used to ensure accurate outputs and the new model will be validated against results from the existing SEE-Segment model. Potential measurments of success unclude segmentation output and quality as well as runtime and reproduceability. 
The primary success criterion will be that the alternative genetic processing library can be integrated successfully and that the resulting implementation can match or exceed the benchmarks created by the existing model.
## Anticipated Challenges

List the main challenges you anticipate and how you might respond to them.

There are two main challenged I anticipate in this project.
1. It may be difficult to get the current SEE-Segment model running, this is where studying the existing documentation and asking the developer, Dirk, for help will be useful.
2. The second problem I foresee is potentially mismatches between the old and new genetic search libraries. It may be that their interfaces work very differently and the code has to be significantly modified. If this is the case we could try helper functions to bridge the gap or if it is extreme enough we could even move to a different library to implement.

## Expected Outcomes

I will consider this project a success if I am able to get the current software running and then implement the new library, resulting in both improved results and better software design.
