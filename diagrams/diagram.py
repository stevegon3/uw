from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
with Diagram("My Architecture"):
    web = EC2("Web Server")
    db = RDS("Database")
    web >> db