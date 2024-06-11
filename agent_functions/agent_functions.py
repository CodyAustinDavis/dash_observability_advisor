##### File to store all the constrained functions that the LLM response will eventually call. 
import re
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql, jobs
from typing import List
import numpy as np



#### Fucntion to create a query, alert, and job to build the alert
def create_alert_and_job(dbx_client: WorkspaceClient, 
                         warehouse_id: str, 
                         alert_name: str, 
                         alert_query: str,
                         alert_schedule: str = None, 
                         subscribers : List[str] = None):


    def convert_to_quartz_cron(standard_cron):
        """
        Convert a standard cron expression to a Quartz cron expression.

        :param standard_cron: str, a standard cron expression with 5 fields
        :return: str, a Quartz cron expression with 7 fields
        """
        # Split the standard cron expression into its components
        fields = standard_cron.split()
        
        # Ensure the standard cron has exactly 5 fields
        if len(fields) != 5:
            raise ValueError("Standard cron expression must have exactly 5 fields")

        # Add the seconds field (0) at the beginning and a placeholder (?) for the day-of-week field
        quartz_cron = f"{fields[0]} {fields[1]} {fields[2]} {fields[3]} {fields[4]} ?"

        return quartz_cron
    

    result = {}
    clean_alert_name = re.sub(r'[^a-zA-Z0-9]', '', alert_name)
    clean_subs = list(np.array(subscribers).flatten())
    clean_cron = convert_to_quartz_cron(alert_schedule)
    dbx_client = dbx_client

    all_warehouses = dbx_client.data_sources.list()

    active_warehouse = [obj.__dict__ for obj in all_warehouses if obj.warehouse_id == warehouse_id][0]
    active_warehouse_data_source = active_warehouse.get('id')


    ### Query Result Object
    query = dbx_client.queries.create(name=f'{clean_alert_name}-DBX-DASH-ALERT-ADVISOR',
                            data_source_id=active_warehouse_data_source,
                            description="Dash Observability Advisor Alert - LLM Generated",
                            query=str(alert_query))
    
    ## Add created query id
    result['query_id'] = query.id

    new_alert = dbx_client.alerts.create(options=sql.AlertOptions(column="alert_condition",
                                                            op="==",
                                                            value="true"),
                            name=f'{clean_alert_name}-DBX-DASH-ALERT-ADVISOR',
                            query_id=query.id
                        )

    ## Add created alert id
    result['alert_id'] = new_alert.id


    sql_task_alert = jobs.SqlTaskAlert(
        alert_id=str(new_alert.id),
        subscriptions=[
            jobs.SqlTaskSubscription(user_name=str(sub)) for sub in clean_subs
        ]
    )

    # Define the SQL Task
    sql_task = jobs.SqlTask(
        alert=sql_task_alert,
        warehouse_id=str(warehouse_id)
    )

    new_job = dbx_client.jobs.create(name=f"DBX-DASH-ALERT-ADVISOR-{clean_alert_name}",
        max_concurrent_runs=1,
        timeout_seconds=0,
        schedule=jobs.CronSchedule(
        quartz_cron_expression="0 0 0 * * ?",
        timezone_id="UTC"
        ),
        queue=jobs.QueueSettings(enabled=True),
        tasks=[
            jobs.Task(description=str(clean_alert_name),
                sql_task=sql_task,
                task_key=str(clean_alert_name),
                timeout_seconds=0)
                ])
    
    

    result['job_id'] = new_job.job_id

    return result



#### Delete and existing alert and job
def delete_alert_and_job(dbx_client: WorkspaceClient, query_id: str, alert_id: str, job_id: str):
    """
    Deletes the specified query, alert, and job in Databricks.

    :param dbx_client: The Databricks WorkspaceClient instance.
    :param query_id: The ID of the query to delete.
    :param alert_id: The ID of the alert to delete.
    :param job_id: The ID of the job to delete.

    """
    try:

        # Delete the job
        if job_id:
            if len(job_id) > 1:
                dbx_client.jobs.delete(job_id)
                print(f"Job with ID {job_id} deleted successfully.")
        
        if alert_id:
            # Delete the alert
            if len(alert_id) > 1:
                dbx_client.alerts.delete(alert_id)
                print(f"Alert with ID {alert_id} deleted successfully.")
                 
        # Delete the query
        if query_id:
            if len(query_id) > 1:
                dbx_client.queries.delete(query_id)
                print(f"Query with ID {query_id} deleted successfully.")
            
    except Exception as e:
        print(f"An error occurred while deleting resources: {str(e)}")

    return


#### This function is designed to parse the results from an LLM (DBRX/Llama3) to create a dictionary of the following: 
def parse_query_result_json_from_string(input_string):
     # Regular expression pattern to match JSON-like structures
    json_pattern = re.compile(r'(\{.*?\})', re.DOTALL)
    
    # Search for the JSON part in the input string
    match = json_pattern.search(input_string)
    
    if match:
        # Extract the JSON part
        json_str = match.group(1)
        
        # Regular expressions to capture specific values without requiring spaces
        alert_name_pattern = query_pattern = re.compile(r'"ALERT_NAME":\s*"([^"]+)"')
        query_pattern = re.compile(r'"QUERY":\s*"([^"]+)"')
        schedule_pattern = re.compile(r'"SCHEDULE":\s*"([^"]+)"')
        recipients_pattern = re.compile(r'"RECIPIENTS":\s*\[([^\]]+)\]')
        context_sql_pattern = re.compile(r'"CONTEXT_SQL":\s*\[([^\]]*)\]')
        
        alert_name = alert_name_pattern.search(json_str)
        query = query_pattern.search(json_str)
        schedule = schedule_pattern.search(json_str)
        recipients = recipients_pattern.search(json_str)
        context_sql = context_sql_pattern.search(json_str)
        

        alert_name_value = alert_name.group(1) if alert_name else None
        query_value = query.group(1) if query else None
        schedule_value = schedule.group(1) if schedule else None
        recipients_value = [email.strip().strip('"') for email in recipients.group(1).split(',')] if recipients else []
        context_sql_value = [sql.strip().strip('"') for sql in context_sql.group(1).split(',')] if context_sql and context_sql.group(1) else []
        
        result = {
            'ALERT_NAME': alert_name_value,
            'QUERY': query_value,
            'SCHEDULE': schedule_value,
            'RECIPIENTS': recipients_value,
            'CONTEXT_SQL': context_sql_value
        }
    else:
        # If no match is found, return an empty dictionary
        result = {
            'ALERT_NAME': None,
            'QUERY': None,
            'SCHEDULE': None,
            'RECIPIENTS': [],
            'CONTEXT_SQL': []
        }
    
    return result
