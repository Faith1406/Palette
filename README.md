# Palette

## *Introduction*

As artificial intelligence continues to evolve, the concept of multi-agent systems—where multiple intelligent agents collaborate to solve tasks—has gained significant traction. When powered by Large Language Models (LLMs) such as OpenAI's GPT or Anthropic's Claude, these agents can perform remarkably human-like reasoning, discussion, and delegation.

Frameworks like AutoGen by Microsoft offer the tools needed to create such ensembles of LLM-powered agents in a conversation-like setup. These agents can take on various roles (e.g., Developer, Reviewer, Analyst) and work together toward a shared goal.

However, while AutoGen is powerful, building multi-agent systems with it typically involves writing a lot of boilerplate code. Users must manage individual agents, assign them system messages, choose LLM providers, set response limits, define stopping conditions, and coordinate interactions. For every experiment or prototype, much of this must be reconfigured from scratch—making it hard to iterate quickly or scale up solutions. To tackle this, we use Palette.

## *Installation*

```bash
pip install Palette
## *Performance Analysis*

Palette is designed with efficiency, modularity, and scalability in mind. By building on top of Microsoft’s AutoGen, it inherits a solid foundation while removing much of the friction involved in multi-agent orchestration. Below is a breakdown of how Palette performs across key dimensions:

### 1. Reduced Development Time

Before Palette: Developers spent considerable time configuring each agent—setting models, prompts, termination logic, and interaction rules manually.

With Palette: Setting up a multi-agent team takes only a few lines of code using simplified constructors and predefined configurations.

*Result:*

- Up to 70–80% reduction in initial setup time.
- Drastically faster iteration speed when testing new ideas or refining agent roles.

---

### 2. Optimized Execution Flow

 includes a built-in conversation orchestrator that manages agent interactions more efficiently. It handles:

- Parallel processing for non-dependent agents
- Intelligent message routing to avoid redundant exchanges
- Early stopping once a satisfactory result is achieved

*Result:*

- Fewer token exchanges, reducing LLM API costs
- Faster conversation cycles, especially in scenarios with 3+ agents

---

### 3. Scalability

AutoAgent supports plug-and-play agent expansion—teams of 3+ agents can be coordinated without introducing extra boilerplate. It allows dynamic agent spawning or retirement based on conversation context or logic rules.

*Result:*

- Smooth scaling from small experiments to complex multi-agent simulations.
- Maintains consistent performance across varying agent counts.

---

### 4. Model Flexibility

AutoAgent supports multiple LLM backends (OpenAI, Anthropic, Azure, etc.) and enables runtime switching. You can assign different agents to different models based on:

- Token efficiency  
- Response speed  
- Role criticality  

*Result:*

- Better resource utilization by mixing lightweight models for simple roles and powerful models for core logic agents.
- Ability to benchmark LLMs within the same conversation setup.

##*Architecture*##
![alt text](<WhatsApp Image 2025-04-07 at 22.05.37_c7c84b01.jpg>)
Palette Core is the central controller that initializes two agents—Primary LLM (task performer) and Critic LLM (reviewer). These agents interact using a Round-Robin Conversation Engine, taking turns to exchange messages. The Conversation Engine manages the dialogue flow and context. It keeps the interaction going until a Termination Condition is met—either through an ExternalTrigger (manual/programmed) or a MentionTrigger (like the critic saying “Approved”). The final result is displayed via the Output Console (terminal, log, or dashboard). This setup enables intelligent, role-based collaboration between agents with minimal setup and clean modularity.