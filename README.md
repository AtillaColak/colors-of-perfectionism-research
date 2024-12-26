# mandala-perfectionism
Main code for calculating is `algining_papers.py`. 
I've tried to comment that code step by step to explain what I tried to do but just in case, below is a summary:
* align the shapes, identify regions on the template, and based on their location, find the corresponding pixels on the colored mandala and compute overflow based on neighboring pixel values compared with the mean of the region.

A problem with this one could be if the scanned image is very distorted or uniquely-shaped, the matching and resizing could damage the calculations. 

## EXAMPLE RESULT
I also included 2 example scanned mandalas with one having clearly more overflow than the other `40033 has more overflow than 40108`. 
The result based on our code somehow indicates that properly: 

```
Overflow Percentage of 40033: 0.32% (28801 pixels overflow)
Overflow Percentage of 40108: 0.25% (22691 pixels overflow)
```

The percentages look promising but I'm not sure. This was a code I came up with after a few unsuccessful attempts, going through some tutorials and stackoverflow. 
