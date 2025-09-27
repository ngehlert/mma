This was the result of a 1 day agentic AI hackathon in September 2025.

It was roughly 4 hours of (vide) coding.

The idea was to setup a project that makes us of agentic AI agents that use AWS strands agents, bedrock and are deployed to AgentCore.

My idea was to have an application, that has an Image creation agent, that can take Pictures with a webcam and analyze those pictures.

Based on the image analysis (in the first state just the group size) it would combine this information with things like weather and time and provide activity suggestions.

For certain activities like books or movies it would make use of another agent to fetch detailed information like current movies in the local cinema or book suggestions for specific genres from Amazon.

This was not modified beyond the code freeze at the hackathon ;) So take the "quality of code" with a grain of salt ;) And the main focus was to get the different agents communicating with each other and not the level of complexity for each agent.  
And to have everything deployed in the end.

## Deployment
```
agentcore launc
```

To invoke the agent in prod you would use 
```
agentcore invoke '{"prompt": "What can I do?"}
```

To run it locally you can either use the `--local` argument from AgentCore or you can just directly invoke `python agent.py`
