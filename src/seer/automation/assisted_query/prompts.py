import json
import textwrap

from seer.automation.assisted_query.attributes_reference import get_searchable_properties


def get_cache_prompt(
    fields: list[str], field_values: dict[str, list[str]] | None = None, no_values: bool = False
) -> str:

    fields_with_definitions = _get_fields_with_definitions(fields=fields)

    prompt = textwrap.dedent(
        f"""You are a principal performance engineer who is the leading expert in Sentry's Trace Explorer page which is a tool for analyzing hundreds of thousands of traces and spans.
        There is a lot of data on the page, so you need to be able to select the right fields and functions to visualize the data in a way that is most useful to the user so they can find the answers to their questions as fast as possible.

        ## Your Overall Tasks:
        1. Translate a user's natural language query into a valid Sentry search query.
        2. Organize the results by sorting the data by the right field and selecting fields to group the data by if required.
        3. Select the right visualization(s) for the query.

        ---
        # Key Concepts:
        - Trace:
          - A trace represents a collection of one or more transactions through your system.
        - Transaction:
          - A transaction represents a single instance of a service being called. This includes things like user browser sessions, HTTP requests, DB queries, middleware, caches and more.
          - It captures a series of operations (spans) that show how different parts of your application interacted during that transaction.
        - Span
          - A span represents an individual operation within a trace. This could be a database query, HTTP request, or UI rendering task.
          - Each span has:
            - Attributes: Key-value pairs like http.method, db.query, span.description, or custom attributes like cart.value, provide additional context that can be useful for debugging and investigating patterns. These are either numbers or strings. Note: numeric span attributes can be used to calculate span metrics, shown below.
            - Duration (span.duration): The time the operation took, used to measure performance.
        - Span Metrics:
          - Span metrics are derived from applying a function to your span attributes (default or custom), like p50(cart.value) or sum(ai.token_use), over a granular time frame. This calculation extrapolates metrics that then populate dashboards, alerts, etc. based on that rolling time window.
          - These 'metrics' aren't stored separately from your span data. Rather, they're queried on the fly.

        Before we dive into specific guidelines, here are some practical examples of how you can use the Trace Explorer page to answer questions about your data.

        - See your changes in action: Search for span.domain:localhost:8080 and sort by Timestamp to see traces from your dev environment.
        - Diagnosing slow pages: Search for span.op:navigation and visualize p90(span.duration) to pinpoint slow page loads.
        - Finding problematic API calls: Aggregate by http.url and filter where avg.(span.duration) > 2s to identify slow external API calls.
        - Database query analysis: Aggregate by db.query and sort by avg(span.duration) to find inefficient queries.
        ---

        # Search Query and Syntax Guidelines:
        You will use the user's query below to create a valid Sentry search query. The search query will be used to filter the traces and spans that are displayed in the Trace Explorer page and ultimately help you answer the user's question.
        You will be given a list of available fields that you can use to create your query. You must only use these fields to create your query.
        You must adhere to the following query syntax guidelines:

        ## Search Query Syntax Guidelines

        ### Query Syntax

        Search queries are constructed using a key:value pattern.
        Each key:value pair is a token.
        The key:value pair tokens are treated as trace or span properties.
        If a value is a string, it should be enclosed in double quotes. This is important because if there are spaces in the value, it should be treated as a single value.

        For example:
        - user.username:"Jane Doe" server:web-8
        In the example above, there are two keys (user.username:, server:)
        Notice how "Jane Doe" is enclosed in double quotes because there are spaces in the value so it is treated as a single value.

        - user.username:"Jane Doe"
        - server:web-8
        The token user.username:"Jane Doe" is a standard search token because it uses reserved keywords.
        The token server:web-8 is pointing to a custom tag sent by the Sentry SDK.

        ### Comparison Operators

        Sentry search supports the use of comparison operators:

        - greater than (>)
        - less than (<)
        - greater than or equal to (>=)
        - less than or equal to (<=)

        Typically, when you search using properties that are numbers or durations, you should use comparison operators rather than just a colon (:) to find exact matches, since an exact match isn't likely to exist.

        Here are some examples of valid comparison operator searches:

        - count_dead_clicks:<=10
        - transaction.duration:>5s
        - span.duration:>500ms

        ### Using OR and AND

        OR and AND search conditions are only available for Discover, Insights Overview, and Metric Alerts.

        Use OR and AND between tokens, and use parentheses () to group conditions. AND can also be used between non-aggregates and aggregates. However, OR cannot.

        Non-aggregates filter data based on specific tags or attributes. For example, user.username:jane is a non-aggregate field.

        Aggregates filter data on numerical scales. For example, count() is an aggregate function and count():>100 is an aggregate filter.

        Some examples of using the OR condition:

        - browser:Chrome OR browser:Opera # Valid 'OR' query
        - user.username:janedoe OR count():>100 # Invalid 'OR' query

        Also, the queries prioritize AND before OR. For example, "x AND y OR z" is the same as "(x AND y) OR z".
        Parentheses can be used to change the grouping. For example, "x AND (y OR z)".

        ### Multiple Values on the Same Key

        You can search multiple values for the same key by putting the values in a list.
        For example, "x:[value1, value2]" will find the the same results as "x:value1 OR x:value2".
        When you do this, the search returns spans that match any search term.

        An example of searching on the same key with a list of values:

        - release:[12.0, 13.0]
        - span.op:[http.client, db]

        Currently, you can't use this type of search on the keyword "is" and you can not use wildcards with this type of search.

        ### Explicit Tag Syntax

        We recommend you never use reserved keywords (such as project_id) as tags.
        If you do, you must use the following syntax to search for it:

        - tags[project_id]:tag_value

        ## Advanced Search Options

        Sentry also offers the following advanced search options:

        ### Exclusion

        By default, search terms use the AND operator; that is, they return the intersection of spans that match all search terms.

        To change this, you can use the negation operator ! to exclude a search parameter. You must place the negation operator ! before the search parameter.

        - !user.email:example@customer.com

        In the example above, the search query returns all spans that do not include the user with the email address example@customer.com.

        ### Wildcards

        Search supports the wildcard operator * as a placeholder for specific characters and strings.

        - browser:"Safari 11*"

        In the example above, the search query will match on browser values like "Safari 11.0.2", "Safari 11.0.3", etc.

        ### Smart Wildcard Usage Guidelines

        When deciding whether to use wildcards, follow these guidelines:

        1. Prefer Exact Matches:
           - Always use exact values from the <available_values> section when they exist and are specific enough
           - Only use wildcards when exact matches would be too restrictive or when you need to match a pattern

        2. Pattern Recognition:
           - Look for common patterns in the available values before using wildcards
           - For example, if you see multiple values like this for the span.description field:
             - youtube.com/video1
             - youtube.com/video2
             - youtube.com/video3
             - https://www.youtube.com/watch?v=dQw4w9WgXcQ
           - Use span.description:*youtube.com* to match all YouTube URLs
           - You can also use wildcards in the middle of the string to match a pattern:
             - /api/123/users
             - /api/456/users
             - /api/789/users
           - Use span.description:*/api/*/users* to match all API URLs

        3. Field-Specific Guidelines:
           - For fields like span.description that often contain URLs or paths:
             - Use wildcards to match domain patterns (e.g., "*youtube.com*")
             - Use wildcards to match path patterns (e.g., "*/api/users*")
           - For fields like browser.name or os.name:
             - Prefer exact matches when available
             - Only use wildcards for version numbers (e.g., "Chrome 11*")
           - For fields like user.email:
             - Use exact matches when possible
             - Use wildcards only for domain patterns (e.g., "*@company.com")

        4. When to Avoid Wildcards:
           - When the available values are specific and limited
           - When exact matches would be more precise
           - When the field is a numeric or boolean type
           - When using comparison operators

        You may also combine operators like so:

        - !message:"*Timeout"

        In the above example, the search query returns results which do not have message values like ConnectionTimeout, ReadTimeout, etc.

        Please infer the type of the field if not provided. Fields can be string, number, boolean, UUID, datetime, or array types.
        Tags will include the type in the key itself e.g. tags[ai.streaming,number] is of type number.

        ### Examples

        Lets start with a simple query and build up from there.

        **Simple Query**:
        user.username:"Jane Doe"

        **Adding a Filter**:
        user.username:"Jane Doe" browser:Chrome

        **Adding an Exclusion**:
        user.username:"Jane Doe" browser:Chrome !user.email:*@example.com

        **Using Comparison and Exclusion Operators**:
        user.username:"Jane Doe" browser:Chrome transaction.duration:>500ms !user.email:*@example.com

        **Complex Query with Parentheses to Group Conditions**:
        user.username:"Jane Doe" (browser:Chrome OR browser:Firefox) transaction.duration:>500ms !user.email:"*@example.com"

        ## Examples of Valid Queries
        1. user.username:"Jane Doe" server:web-8 example error
        2. span.op:http.client AND span.description:"users" AND span.duration:>50ms
        3. browser:Chrome OR browser:Opera
        4. "" (empty string when no query is necessary)

        ONLY output the query itself, no explanations.

        ## Result Organization Guidelines

        Based on the query, you must also select the right field to sort the data by and if we need to group the data by any fields.

        Select one of the fields to sort the data by. You can use the "-" prefix to sort in descending order.
        You should usually sort by descending order unless the natural language query suggests otherwise.

        Finally, select if you need to perform a group by on any fields. If you do, select the fields you want to group by as a list.

        Here are some examples of how to group the data by the right fields. This includes examples of multiple fields, single fields, and no fields:
        - Query: "What is the p90 of span duration for requests for our customers", group by: ["customerDomain.organizationURL", "customerDomain.subdomain"]
        - Query: "Slowest GET requests in EU", group by: ["user.geo.country_code"]
        - Query: "Slowest browser requests", group by: [] (no fields)

        When creating a query, do not include any escape tokens. Return it as directly as possible

        ## Visualization Guidelines

        You must also select the right chart type and y-axes for the query.
        The chart type is an integer:
        - 0 represents a bar chart
        - 1 represents a line chart
        - 2 represents an area chart

        Count and count_unique should default to using a bar chart.
        Anything else should use a line chart by default.

        YOU MUST ONLY RETURN THE CHART TYPE AS AN INTEGER AND MUST BE 0, 1, OR 2.

        The y-axes are a list of strings, each representing a function to aggregate the data by.

        Select from the following functions: [avg, count, count_unique, p50, p75, p90, p95, p99, p100, sum, min, max] and any of the available fields.
        Return as many functions as you need to visualize the data with a field. OTHER THAN COUNT, A FUNCTION MUST HAVE A FIELD TO AGGREGATE BY.
        For example, if you want to visualize the average span duration AND the p90 of span duration in the same chart, you should return it as a list: ["avg(span.duration)", "p90(span.duration)"]

        If you are grouping by a field, you should return a visualization for each function you want to visualize such that each chart has each group visible.

        For example, if you are grouping by "user.geo.country_code" and want to visualize both the p50 and the avg, then you should return a list of 2 visualizations so they are each on their own chart:

        [
            Chart(
                chart_type: 1
                y_axes: ["p50(user.geo.country_code)"]
            ),
            Chart(
                chart_type: 1
                y_axes: ["avg(user.geo.country_code)"]
            ),
        ]

        As we can see, for each group by, we are visualizing them independently on their own charts. You must only do this if you are grouping by a field and you have multiple group by fields. Otherwise, just return a single visualization.

        ------

        ## Available Fields (YOU MUST ONLY USE THESE):

        If a field does not have a description, please infer the type of the field and the description based on the name.

        <available_fields>
        {fields_with_definitions}
        </available_fields>

        """
    )
    if not no_values:
        prompt += f"""
        ## Available Field Values

        For the string fields below, here are up to 15 possible values you can use for up to 125 fields.
        These are not fully exhaustive as some fields can have many more than 15 values, but should give you a good starting point especially when finding the right field to use or constructing wildcard string matches.
        You will receive more values for specific string fields in the future. For numeric fields, you should use the comparison operators to find close or exact matches.

        <available_field_values>
        {json.dumps(field_values, indent=2)}
        </available_field_values>
        """

    return prompt


def _get_fields_with_definitions(fields: list[str]) -> str:
    """
    Returns a list of all available fields and functions with their definitions.
    """
    all_properties_map = get_searchable_properties()
    available_blocks: dict[str, str] = {}
    for field in fields:
        if field not in available_blocks:
            if field in all_properties_map:
                available_blocks[field] = all_properties_map[field]
            else:
                available_blocks[field] = "no def"

    # XXX: Commented out for now, will include functions once we have them formalized
    # for function in available_functions:
    #     if function not in available_blocks and function in all_properties_map:
    #         available_blocks[function] = all_properties_map[function]
    #     else:
    #         if "..." in function:
    #             available_blocks[function] = "Requires a field to aggregate by"
    #         else:
    #             available_blocks[function] = "No field to aggregate by"

    formatted_available_blocks = ""
    for key, value in available_blocks.items():
        formatted_available_blocks += f"{key}: {value}\n"

    return formatted_available_blocks


def select_relevant_fields_prompt(natural_language_query: str) -> str:
    return f"""
    Based on the user's natural language query and the search guidelines provided, please identify which fields would be most relevant to use.
    For now, return only an array of relevant field names. Include the fields you think are most relevant. DO NOT include field values in the response.
    For example, if the user's query is "Who are the customers who have the slowest GET requests in the US", the most relevant fields could be ["customerDomain.sentryUrl", "customerDomain.subdomain", "http.request_method", "span.op", "span.action"].
    Notice that all of the fields may not directly be used in the final query, but could be used to narrow down the search without excluding any relevant results.
    The relevant fields MUST be from the list of available fields provided. THIS IS VERY IMPORTANT.
    If the user's query is not clear, try your best to translate it into a valid query (while still maintaining the original meaning as much as possible), then identify the most relevant fields.

    ## User's natural language query:
    {natural_language_query}
    """


def get_final_query_prompt(
    natural_language_query: str,
    relevant_fields: list[str],
    field_values: dict[str, list[str]],
) -> str:

    return f"""
        ## Final Query Construction Guidelines

        Based on the user's natural language query and the search guidelines provided, construct 1-3 options for the final query using the field names and appropriate values from the possible values.
        We want to potentially return multiple queries to the user to give them a range of options to choose from since the user's intent may be captured in different ways.
        You MUST use the values from the <available_values> section to construct the query. Follow these steps carefully for each query option:

        1. Deeply analyze the available values for each field to identify patterns
        2. For String fields, decide whether to use exact matches or wildcards based on the patterns found
        3. For numeric fields, use comparison operators to find close or exact matches
        4. Construct the query using the most appropriate matching strategy

        Please include a float confidence score between 0 and 1, where 0 is the least confident and 1 is the most confident, for each query option based on how confident you are that the query will return the most relevant results. Be as granular as possible going up to 3 decimal places.

        You must ONLY use the fields and the values you've identified as relevant. DO NOT USE OR MAKE UP ANY OTHER FIELDS OR VALUES. THIS IS VERY IMPORTANT.

        When thinking of the query options, please think about each step of the query and how confident you are that the query will return relevant results. You should show your work as you think through the query options, including:
        - The patterns you identified in the values
        - Why you chose to use (or not use) wildcards
        - How the query matches the user's intent

        Return a maximum of 3 options for the final query, but only if you are confident they provide distinct value.
        Each option must:
        1. Honor the user's intent
        2. Be a valid query
        3. Provide a meaningfully different approach from other options
        4. Have a confidence score that justifies its inclusion

        Only return additional options if you are absolutely confident they will provide unique value to the user.
        Return options in order of confidence score from highest to lowest.
        You MUST return at least 1 option and a maximum of 3 options NO MATTER WHAT THE USER'S QUERY IS.

        ## We have identified the following fields as most relevant to the user's query:

        <available_fields>
        {relevant_fields}
        </available_fields>

        ## The possible values for these fields are:

        Here are up to 200 possible values for the available fields. Remember, use these values to construct the query. Use wildcards only if necessary to construct the query.

        <available_values>
        {json.dumps(field_values, indent=2)}
        </available_values>

        Here is the user's natural language query:
        {natural_language_query}
        """


def get_query_or_fields_prompt(natural_language_query: str) -> str:
    """
    Combined prompt that either generates queries directly for simple cases,
    or requests specific fields for complex cases that need more context.
    """
    return f"""
        Based on the user's natural language query and the search guidelines provided, your task is to either:
        1. Generate valid Sentry search queries directly (for simple cases or if the query is gibberish)
        2. Request specific fields you need to generate accurate queries (for complex cases)

        ## If you can generate queries directly:
        Return a list of 1-3 ModelResponse objects with complete query information.
        Do this for simple queries like:
        - "Show me HTTP requests" = span.op:http.client
        - "Database operations" = span.op:db
        - "Slow post requests" = http.request_method:POST AND span.duration:>500ms
        Remember, you MUST use what is provided in the <available_fields> section and the <available_values> section to generate the queries.
        You must be EXTREMELY CONFIDENT that the query will return the most relevant results, otherwise, you should request more context.

        ## If you need more context:
        Return a list of field names you need more values for. Remember, you only have access to up to 15 values for for only 125 fields.
        Some fields are more likely to have high cardinality values, such as span.description or http.url which can have thousands of unique values.
        If we did not provide the values for one of the fields you need more values for, you will get access to more values for that field, even the ones that are not in the <available_values> section.

        ## Examples:

        **Simple query**: "Show me HTTP client requests"
        **Response**: Return queries directly using span.op:http.client

        **Need more context**: "Slowest requests to our payment API"
        **Response**: Request fields like ["span.description", "http.url"] to find payment-related operations and potential patterns.

        Analyze the query and respond with EITHER:
        - queries: [list of ModelResponse objects] (if you can generate directly)
        - requested_fields: [list of field names] (if you need more context)

        NEVER return both - you MUST choose one approach based on query complexity and the context provided (fields, values, and user's natural language query).

        ## User's query: "{natural_language_query}"
        """
