from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

args = {
    'owner': 'ailab',
    'depends_on_past': False, 
    'email': [''], 
    'email_on_failure': False,
    'email_on_retry': False, 
    'retries': 1, 
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='fifa_batch_process',
    default_args=args,
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=1),
    schedule_interval='@hourly',
    catchup=False,
)

templated_command = """
python3 /home/jovyan/work/mhj/fifa/src/FIFAOnline4.py -t batch_matchdetail
"""

# [START howto_operator_bash]
run_this = BashOperator(
    task_id='bash_test',
    bash_command=templated_command,
    dag=dag,
)
# [END howto_operator_bash]

if __name__ == "__main__":
    dag.cli()