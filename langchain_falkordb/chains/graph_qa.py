"""Question answering over a FalkorDB graph by generating Cypher."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig

from langchain_falkordb.chains.prompts import (
    CYPHER_GENERATION_PROMPT,
    CYPHER_QA_PROMPT,
)
from langchain_falkordb.graphs import FalkorDBGraph

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

_CYPHER_FENCE_PATTERN = re.compile(r"```(?:cypher)?(.*?)```", re.DOTALL)


def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from. The code may be wrapped in
            triple backticks, with or without a ``cypher`` language tag.

    Returns:
        The extracted Cypher statement.
    """
    matches = _CYPHER_FENCE_PATTERN.findall(text)
    return matches[0].strip() if matches else text.strip()


class FalkorDBQAChain(Runnable[Union[str, Dict[str, Any]], Dict[str, Any]]):
    """Chain for question-answering against a FalkorDB graph by generating
    Cypher statements.

    The chain is a ``Runnable``: invoke it with ``{"query": question}`` (or
    a plain question string) and it returns ``{"result": answer}``.

    Example:
        .. code-block:: python

            from langchain_falkordb import FalkorDBGraph, FalkorDBQAChain
            from langchain_openai import ChatOpenAI

            graph = FalkorDBGraph("movies")
            chain = FalkorDBQAChain.from_llm(
                ChatOpenAI(model="gpt-4o-mini"),
                graph=graph,
                allow_dangerous_requests=True,
            )
            chain.invoke({"query": "Who acted in Forrest Gump?"})

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the
        calling code may attempt commands that would result in deletion,
        mutation of data if appropriately prompted or reading sensitive data
        if such data is present in the database.
        The best way to guard against such negative outcomes is to (as
        appropriate) limit the permissions granted to the credentials used
        with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        *,
        graph: FalkorDBGraph,
        cypher_generation_chain: Runnable[Dict[str, Any], str],
        qa_chain: Runnable[Dict[str, Any], str],
        input_key: str = "query",
        output_key: str = "result",
        top_k: int = 10,
        return_intermediate_steps: bool = False,
        return_direct: bool = False,
        allow_dangerous_requests: bool = False,
    ) -> None:
        """Initialize the chain.

        Args:
            graph: The FalkorDB graph to query.
            cypher_generation_chain: Runnable that turns
                ``{"question", "schema"}`` into a Cypher statement.
            qa_chain: Runnable that turns ``{"question", "context"}`` into
                the final answer.
            input_key: Key of the question in the input dict.
            output_key: Key of the answer in the output dict.
            top_k: Maximum number of query result rows to pass to the
                QA prompt.
            return_intermediate_steps: Also return the generated Cypher and
                the raw query context under ``intermediate_steps``.
            return_direct: Return the raw query result instead of running
                the QA prompt.
            allow_dangerous_requests: Required opt-in acknowledging that the
                chain can send arbitrary generated Cypher to the database.
        """
        if allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can "
                "make dangerous requests by setting `allow_dangerous_requests` "
                "to `True`. You must narrowly scope the permissions of the "
                "database connection to only include necessary permissions. "
                "Failure to do so may result in data corruption or loss or "
                "reading sensitive data if such data is present in the "
                "database. Only use this chain if you understand the risks "
                "and have taken the necessary precautions. See "
                "https://python.langchain.com/docs/security for more "
                "information."
            )
        self.graph = graph
        self.cypher_generation_chain = cypher_generation_chain
        self.qa_chain = qa_chain
        self.input_key = input_key
        self.output_key = output_key
        self.top_k = top_k
        self.return_intermediate_steps = return_intermediate_steps
        self.return_direct = return_direct

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        graph: FalkorDBGraph,
        cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
        qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
        **kwargs: Any,
    ) -> FalkorDBQAChain:
        """Initialize the chain from an LLM.

        Args:
            llm: The language model used both to generate Cypher and to
                phrase the final answer.
            graph: The FalkorDB graph to query.
            cypher_prompt: Prompt used to generate the Cypher statement.
            qa_prompt: Prompt used to generate the final answer.
            kwargs: Additional arguments passed to the constructor.
        """
        return cls(
            graph=graph,
            cypher_generation_chain=cypher_prompt | llm | StrOutputParser(),
            qa_chain=qa_prompt | llm | StrOutputParser(),
            **kwargs,
        )

    def invoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a Cypher statement, run it and answer the question."""
        if isinstance(input, str):
            question = input
        else:
            question = input[self.input_key]

        generated_cypher = extract_cypher(
            self.cypher_generation_chain.invoke(
                {"question": question, "schema": self.graph.get_schema},
                config=config,
            )
        )

        intermediate_steps: List[Dict[str, Any]] = [{"query": generated_cypher}]

        context = self.graph.query(generated_cypher)[: self.top_k]

        if self.return_direct:
            final_result: Any = context
        else:
            intermediate_steps.append({"context": context})
            final_result = self.qa_chain.invoke(
                {"question": question, "context": context},
                config=config,
            )

        chain_result: Dict[str, Any] = {self.output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
        return chain_result
