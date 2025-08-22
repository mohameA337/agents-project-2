from typing import List, Dict
import os
from pydantic import BaseModel, Field, field_validator

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

class KeywordSpec(BaseModel):
    primary_keywords: list[str] = Field(..., min_length=1, description="3–10 core keywords")
    secondary_keywords: list[str] = Field(default_factory=list, description="0–15 supporting keywords")
    audience: str = Field(..., min_length=3, max_length=80)
    tone: str = Field(default="informative", description="informative|persuasive|narrative|technical|casual|formal")

    @field_validator("primary_keywords", "secondary_keywords")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        """Remove duplicates and normalize keywords"""
        seen, out = set(), []
        for k in (x.strip() for x in v if x and x.strip()):
            low = k.lower()
            if low not in seen:
                seen.add(low)
                out.append(k)
        return out

    @field_validator("tone")
    @classmethod
    def _tone(cls, v: str) -> str:
        allowed = {"informative", "persuasive", "narrative", "technical", "casual", "formal"}
        return v if v in allowed else "informative"


@CrewBase
class MyCrew:

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Put it in your .env file.")

        self.llm = LLM(
            model=os.environ["MODEL"],       
            api_key=os.environ.get("GEMINI_API_KEY")
        )

    @agent
    def input_kickoff(self) -> Agent:
        return Agent(config=self.agents_config['input_kickoff'], llm=self.llm, verbose=True)

    @agent
    def keyword_instructor(self) -> Agent:
        return Agent(config=self.agents_config['keyword_instructor'], llm=self.llm, verbose=True)

    @agent
    def story_writer(self) -> Agent:
        return Agent(config=self.agents_config['story_writer'], llm=self.llm, verbose=True)

    @agent
    def evaluator(self) -> Agent:
        return Agent(config=self.agents_config['evaluator'], llm=self.llm, verbose=True)

    @agent
    def reporter(self) -> Agent:
        return Agent(config=self.agents_config['reporter'], llm=self.llm, verbose=True)

    @task
    def kickoff_task(self) -> Task:
        return Task(config=self.tasks_config['kickoff_task'])

    @task
    def keywords_task(self) -> Task:
        return Task(
            config=self.tasks_config['keywords_task'],
            output_pydantic=KeywordSpec
        )

    @task
    def story_task(self) -> Task:
        return Task(config=self.tasks_config['story_task'])

    @task
    def evaluate_task(self) -> Task:
        return Task(config=self.tasks_config['evaluate_task'])

    @task
    def report_task(self) -> Task:
        return Task(config=self.tasks_config['report_task'], output_file='output/content_report.md')

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.input_kickoff(),
                self.keyword_instructor(),
                self.story_writer(),
                self.evaluator(),
                self.reporter(),
            ],
            tasks=[
                self.kickoff_task(),
                self.keywords_task(),   
                self.story_task(),
                self.evaluate_task(),
                self.report_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )
