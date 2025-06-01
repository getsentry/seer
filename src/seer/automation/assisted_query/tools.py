from seer.automation.agent.tools import ClaudeTool, FunctionTool
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


class SearchTools:

    def __init__(self, org_id: int, project_ids: list[int]):
        self.org_id = org_id
        self.project_ids = project_ids

    @inject
    def substring_values_search(
        self,
        field: str,
        substring: str,
        stats_period: str = "48h",
        rpc_client: RpcClient = injected,
    ) -> list[str]:
        """
        Searches for values for a given field if it contains a given substring.
        """

        response = rpc_client.call(
            "get_attribute_values_with_substrings",
            org_id=self.org_id,
            project_ids=self.project_ids,
            field=field,
            substring=substring,
            stats_period=stats_period,  # TODO: Should this also support absolute timerange searches?
            limit=1000,
        )

        return (
            response.get("values", []) if response else []
        )  # TODO: Probably should not fail quietly?

    def get_tools(self) -> list[ClaudeTool | FunctionTool]:

        # TODO: Simplify example in the description
        tools: list[ClaudeTool | FunctionTool] = [
            FunctionTool(
                name="substring_values_search",
                fn=self.substring_values_search,
                description='Searches for values for a given field if it contains a given substring. ie: If a field "food" has values ["apple", "cheesesteak", "grilled cheese", "fish and chips", "mac and cheese", "banana", "fries with cheese and onions"]and the substring "cheese" is given, it will return ["cheesesteak", "grilled cheese", "mac and cheese", "fries with cheese and onions"]',
                parameters=[
                    {
                        "name": "field",
                        "type": "string",
                        "description": "The field for which you are requesting values for.",
                    },
                    {
                        "name": "substring",
                        "type": "string",
                        "description": "The substring match you want to search values for.",
                    },
                    {
                        "name": "stats_period",
                        "type": "string",
                        "stats_period": "The timerange relative to now which you want to search for. Defaults to 48h (between now and 48 hours ago)",
                    },
                ],
                required=["field", "substring"],
            )
        ]

        return tools
