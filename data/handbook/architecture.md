# System Architecture Overview

This page gives new engineers a map of the main systems that run the robot fleet.

## Fleet controller

The fleet controller assigns tasks to robots and tracks their state. It is written in Go and runs in each customer's region. Robots keep a persistent gRPC stream open to the fleet controller and send a heartbeat every 2 seconds.

The fleet controller has an availability objective of 99.95% per month, and the p99 latency for sending a command to a robot must stay under 200 milliseconds.

## Route planner

The route planner computes collision-free paths through the warehouse. It is written in Rust for predictable latency. A typical route is planned in under 50 milliseconds; plans that take longer than 10 seconds count as a SEV2 degradation.

## Telemetry pipeline

Robots stream telemetry, such as position, battery level, and error codes, into Kafka. A set of consumers writes the data to ClickHouse for analytics and dashboards. Each robot produces roughly 2 MB of telemetry per hour.

## Customer dashboard

Customers manage their fleet through a web dashboard built with TypeScript and React. It talks to the public API described in the API Guidelines.

## Where things run

Production runs on Kubernetes in three cloud regions: us-west, us-east, and eu-central. Customer data stays in the region the customer chose during onboarding.
