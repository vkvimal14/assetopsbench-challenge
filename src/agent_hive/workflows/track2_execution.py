"""
Track 2: Execution-Oriented Dynamic Multi-Agent Workflow
-------------------------------------------------------

Template for participants to implement a DynamicWorkflow.
This workflow is inspired by SequentialWorkflow, but allows for:
    - Multiple agents per task
    - Dynamic / conditional task execution
    - Parallel task execution (if desired)
    - Helper agents for fallback or support
    - Smarter context propagation across tasks

⚠️ IMPORTANT:
Participants are expected to edit ONLY the clearly marked TODO sections.
All other logic (executor, memory, orchestration) must remain unchanged.
"""

import json
from typing import List
from pydantic import Field

from agent_hive.enum import ContextType
from agent_hive.task import Task
from agent_hive.workflows.base_workflow import Workflow
from agent_hive.logger import get_custom_logger
from agent_hive.agents.base_agent import BaseAgent

logger = get_custom_logger(__name__)

# =========================================================
# 🛠️ EDITABLE SECTION for Participants
# 🎯 Purpose: Implement revision / validation of task inputs
#
# ✅ Allowed:
#   - Define your own revision rules (clarity check, validation, enrichment)
#   - Add formatting styles (bullet points, numbered lists, emojis, etc.)
#   - Suggest metadata (tags, difficulty, clarity score)
#   - Return multiple variants of a revision
#
# ❌ Not Allowed:
#   - Modify workflow execution logic outside this agent
#   - Replace the base ReAct agent or Executor
#   - Change memory persistence or retry logic
# =========================================================

class TaskRevisionHelperAgent(BaseAgent):
    """
    Enhanced TaskRevisionHelperAgent with intelligent task validation and refinement.
    """

    name = "TaskRevisionHelperAgent"
    description = "Revises, validates, and provides suggestions for task inputs with intelligent analysis."
    memory = []
    tools = []

    def __init__(self, llm: str = None, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def execute_task(self, task_input: str) -> str:
        """
        Review and revise the given task input, suggesting improvements or validation.

        Args:
            task_input (str): The task description to revise.

        Returns:
            str: Revised/validated task description or feedback.
        """
        # =========================================================
        # 🚧 TODO: Implement your revision logic here
        # =========================================================
        
        # Enhanced revision logic with validation
        revised_task = task_input.strip()
        
        # Add clarity improvements
        if not revised_task.endswith(('.', '!', '?')):
            revised_task += '.'
        
        # Add context markers if missing
        if 'site' not in revised_task.lower() and 'asset' not in revised_task.lower():
            logger.info("⚠️ Task may benefit from specifying site/asset context")
        
        # Check for sensor/data requirements
        if any(keyword in revised_task.lower() for keyword in ['sensor', 'data', 'history', 'forecast']):
            if 'date' not in revised_task.lower() and 'period' not in revised_task.lower():
                logger.info("💡 Consider adding time period specification for data queries")
        
        # Validate failure mode queries
        if 'failure' in revised_task.lower():
            if 'asset' not in revised_task.lower() and 'equipment' not in revised_task.lower():
                logger.info("🔧 Failure mode queries should specify target equipment")
        
        return revised_task


class DynamicWorkflow(Workflow):
    """
    Enhanced DynamicWorkflow with intelligent agent selection and fallback strategies.
    """

    context_type: ContextType = Field(
        default=ContextType.DISABLED, description="Type of context to use."
    )

    def __init__(
        self,
        tasks: List[Task],
        context_type: ContextType = ContextType.DISABLED,
        max_memory: int = 10,
    ):
        self.tasks = tasks
        self.context_type = context_type
        self.memory: List[str] = []
        self.max_memory = max_memory
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")

        for i, task in enumerate(self.tasks):
            if not task.agents:
                raise ValueError("Task must have at least one agent")

            if len(task.agents) > 1:
                logger.info(
                    f"Task {i+1} has {len(task.agents)} agents. "
                    "DynamicWorkflow must decide how to use them (primary, fallback, parallel, etc.)."
                )

            # Validate context dependencies
            if self.context_type == ContextType.SELECTED and isinstance(task.context, list):
                for context_task in task.context:
                    if context_task not in self.tasks[:i]:
                        raise ValueError(
                            "task.context must be a list of Task objects "
                            "that are part of the workflow"
                        )

    def run(self):
        """
        Execute tasks dynamically with enhanced agent coordination.
        """
        self.memory = []

        # =========================================================
        # 🚧 TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Customize how agents are scheduled and executed
        #
        # ✅ Allowed:
        #   - Replace for-loop with while-loop (max iterations fixed at 15)
        #   - Use TaskRevisionHelperAgent to refine or validate responses
        #   - Experiment with combining multiple agent responses
        #   - Add fallback strategies (e.g., retry with helper agent)
        #
        # ❌ Not Allowed:
        #   - Remove safety cap on iterations (max = 15 must remain)
        #   - Change memory persistence logic
        #   - Replace Executor orchestration logic
        # =========================================================
        
        self.context_type = ContextType.SELECTED
        max_loops = 15
        revision_helper = TaskRevisionHelperAgent()
        
        i = 0
        while i < len(self.tasks) and i < max_loops:
            task = self.tasks[i]
            task_no = i + 1
            
            logger.info(f"📋 Task {task_no}/{len(self.tasks)}: {task.description}")

            assigned_agents = task.agents

            # Build input with context
            user_input = self._build_input(task, i)
            
            # Optional: Revise task input for clarity
            try:
                revised_input = revision_helper.execute_task(user_input)
                if revised_input != user_input:
                    logger.info(f"✨ Task input refined: {revised_input}")
                    user_input = revised_input
            except NotImplementedError:
                # If revision not implemented, continue with original
                pass

            # Execute with primary agent
            try:
                response = assigned_agents[0].execute_task(user_input)
                logger.info(f"✅ Primary agent ({assigned_agents[0].name}) completed")
            except Exception as e:
                logger.error(f"❌ Primary agent failed: {e}")
                
                # Fallback to secondary agent if available
                if len(assigned_agents) > 1:
                    logger.info(f"🔄 Attempting fallback to {assigned_agents[1].name}")
                    try:
                        response = assigned_agents[1].execute_task(user_input)
                        logger.info(f"✅ Fallback agent succeeded")
                    except Exception as fallback_error:
                        logger.error(f"❌ Fallback also failed: {fallback_error}")
                        response = f"Error: Both primary and fallback agents failed"
                else:
                    response = f"Error: {str(e)}"

            # Clean up response
            response = response.replace("Final Answer:", "").strip()
            self.memory.append(response)

            i += 1

        # =========================================================
        # END OF EDITABLE SECTION
        # =========================================================

        history = self.generate_history()
        print(json.dumps(history, indent=4))
        return history

    def _build_input(self, task: Task, idx: int) -> str:
        """Helper to build task input string based on context type."""
        if self.context_type == ContextType.DISABLED:
            return task.description

        elif self.context_type == ContextType.ALL:
            context = "\n".join(self.memory[-self.max_memory :])
            return f"{task.description}\n\nContext:\n{context}"

        elif self.context_type == ContextType.PREVIOUS:
            if not self.memory:
                return task.description
            context = self.memory[-1]
            return f"{task.description}\n\nContext:\n{context}"

        elif self.context_type == ContextType.SELECTED:
            context_tasks = task.context or []
            context = ""
            for context_task in context_tasks:
                idx = self.tasks.index(context_task)
                if idx >= len(self.memory):
                    raise IndexError(
                        f"Context task {context_task.description} not found in memory"
                    )
                context += self.memory[idx] + "\n"
            return f"{task.description}\n\nContext:\n{context}"

        else:
            raise ValueError(f"Invalid context_type: {self.context_type}")

    def generate_history(self):
        """
        Return structured execution history for evaluation.
        """
        history = []
        for i, task in enumerate(self.tasks):
            history.append(
                {
                    "task_number": i + 1,
                    "task_description": task.description,
                    "agent_names": [agent.name for agent in task.agents],
                    "response": self.memory[i] if i < len(self.memory) else None,
                }
            )
        return history
