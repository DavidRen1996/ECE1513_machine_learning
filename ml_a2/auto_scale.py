import boto3
import time

from datetime import datetime, timedelta

from app import config

max_threshold = 70
min_threshold = 20

increase_rate = 1.25
decrease_rate = 0.75

while True:

    # create connection to ec2
    ec2 = boto3.resource('ec2')

    instances = ec2.instances.filter(
      Filters=[
          {'Name': 'placement-group-name',
           'Values': ['A2_workerpool']
           },

          {'Name': 'instance-state-name',
           'Values': ['running']
           },
      ]
    )

    #instances = ec2.instances.all()
    cpu_stats_1 = []
    ids = []

    for instance in instances:

        ids.append(instance.id)

        client = boto3.client('cloudwatch')

        # get cpu statistics in 1 minute(60s)

        cpu_1 = client.get_metric_statistics(
            Period=60,
            StartTime=datetime.utcnow() - timedelta(seconds=2 * 60),
            EndTime=datetime.utcnow() - timedelta(seconds=1 * 60),
            MetricName='CPUUtilization',
            Namespace='AWS/EC2',  # Unit='Percent',
            Statistics=['Average'],
            Dimensions=[{'Name': 'InstanceId',
                         'Value': instance.id}]
        )

        # gather return statistics

        for point in cpu_1['Datapoints']:

            load = round(point['Average'], 2)
            cpu_stats_1.append(load)

    average_load = sum(cpu_stats_1) / len(cpu_stats_1)

    print(cpu_stats_1)
    print(average_load)
    print(ids)

# option 1
    if average_load > max_threshold:
        # the number of new ec2 instances
        add_instance_num = int(len(cpu_stats_1) * (increase_rate-1)+1)
        print(add_instance_num)

        # add instances
        for i in range(add_instance_num) :
            instances = ec2.create_instances(ImageId=config.ami_id, InstanceType='t2.small', MinCount=1, MaxCount=1,
                                             Monitoring={'Enabled': True},
                                             Placement={'AvailabilityZone': 'us-east-1a', 'GroupName': 'A2_workerpool'},
                                             SecurityGroups=[
                                                 'launch-wizard-11',
                                             ],
                                             KeyName='ece1779_A2_user',
                                             TagSpecifications=[
                                                 {
                                                     'ResourceType': 'instance',
                                                     'Tags': [
                                                         {
                                                             'Key': 'Name',
                                                             'Value': 'Additional_workers'
                                                         },
                                                     ]
                                                 },
                                             ]
                                             )


        # resize ELB
        for instance in instances:
            print(instance.id)
            ec2 = boto3.resource('ec2')
            instance.wait_until_running(
                Filters=[
                    {
                        'Name': 'instance-id',
                        'Values': [
                            instance.id,
                        ]
                    },
                ],
            )

            print(instance.id)
            client = boto3.client('elbv2')
            client.register_targets(
                TargetGroupArn='arn:aws:elasticloadbalancing:us-east-1:244202399167:targetgroup/target/05c97e5450467af2',
                Targets=[
                    {
                        'Id': instance.id,
                    },
                ]
            )


            # wait until finish
            waiter = client.get_waiter('target_in_service')
            waiter.wait(
                TargetGroupArn='arn:aws:elasticloadbalancing:us-east-1:244202399167:targetgroup/target/05c97e5450467af2',
                Targets=[
                    {
                        'Id': instance.id,
                    },
                ],
            )


# option 2
    if average_load < min_threshold:
        minus_instance_num = int(len(cpu_stats_1) * (1-decrease_rate))
        print(minus_instance_num)

        if minus_instance_num > 0:
            ids_to_delete = ids[:minus_instance_num]
            print(ids_to_delete)

            #resize ELB
            for id in ids_to_delete:
                print(id)
                client = boto3.client('elbv2')
                client.deregister_targets(
                    TargetGroupArn='arn:aws:elasticloadbalancing:us-east-1:244202399167:targetgroup/target/05c97e5450467af2',
                    Targets=[
                        {
                            'Id': id,
                        },
                    ]
                )

                # wait until finish
                waiter = client.get_waiter('target_deregistered')
                waiter.wait(
                    TargetGroupArn='arn:aws:elasticloadbalancing:us-east-1:244202399167:targetgroup/target/05c97e5450467af2',
                    Targets=[
                        {
                            'Id': id,
                        },
                    ],

                )

            # drop instances
            for id in ids_to_delete:
                print(id)
                ec2.instances.filter(InstanceIds=[id]).terminate()

    # wait for 1 minute
    time.sleep(60)