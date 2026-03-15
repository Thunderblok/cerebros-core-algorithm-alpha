"""Example: Runic-Jido Integrated Neural Architecture Search

Demonstrates using the Runic DAG workflow engine with Jido's signal-driven
agent system to orchestrate Cerebros Neural Architecture Search.

This example shows:
1. Building a NAS workflow as a Runic DAG
2. Running it through a Jido signal-driven agent
3. Collecting results with full provenance tracking
4. Introspecting execution history

The integration ports concepts from:
- Runic (https://github.com/zblanco/runic) - Elixir DAG workflow composition
- JidoRunic (https://github.com/agentjido/jido_runic) - Signal-driven agent bridge
"""

from cerebros.runic import Workflow, Step, Condition, Rule, Pipeline
from cerebros.jido import Signal, SignalType, Fact, ActionNode, AgentLoop, Strategy
from cerebros.jido.introspection import Introspection
from cerebros.runic_jido import (
    NASWorkflowBuilder,
    create_nas_workflow,
    NASAgent,
    GenerateArchitectureAction,
    BuildModelAction,
    TrainModelAction,
    EvaluateModelAction,
)


def example_basic_workflow():
    """Example 1: Basic Runic workflow with Steps and Rules."""
    print("=" * 60)
    print("Example 1: Basic Runic Workflow")
    print("=" * 60)

    # Create a simple workflow with steps
    wf = Workflow(name="basic_pipeline")

    # Step 1: Double a value
    double = Step(name="doubled", func=lambda ctx: ctx.get("input", 0) * 2)
    wf.add_component(double)

    # Step 2: Add 10 (depends on step 1)
    add_ten = Step(
        name="plus_ten",
        func=lambda ctx: ctx.get("doubled", 0) + 10,
    )
    wf.add_component(add_ten, depends_on=["doubled"])

    # Rule: Only square if result > 20
    square_rule = Rule(
        name="maybe_square",
        condition=Condition(
            predicate=lambda ctx: ctx.get("plus_ten", 0) > 20,
        ),
        reaction=Step(
            func=lambda ctx: ctx.get("plus_ten", 0) ** 2,
        ),
    )
    wf.add_component(square_rule, depends_on=["plus_ten"])

    # Visualize the DAG
    print(wf.visualize())
    print()

    # Execute
    result = wf.run({"input": 7})
    print(f"Input: 7")
    print(f"Doubled: {result.context.get('doubled')}")
    print(f"Plus ten: {result.context.get('plus_ten')}")
    print(f"Maybe square: {result.context.get('maybe_square')}")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Execution order: {result.execution_order}")
    print(f"Provenance: {result.provenance}")
    print()


def example_signal_driven_agent():
    """Example 2: Jido signal-driven agent with strategies."""
    print("=" * 60)
    print("Example 2: Signal-Driven Agent")
    print("=" * 60)

    # Create a simple workflow
    wf = Workflow(name="signal_processor")
    process_step = Step(
        name="process",
        func=lambda ctx: {
            "processed": True,
            "signal_type": str(ctx.get("signal_type", "unknown")),
            "data": ctx.get("signal_data"),
        },
    )
    wf.add_component(process_step)

    # Create a strategy that handles COMMAND signals
    strategy = Strategy(
        name="command_handler",
        workflow=wf,
        signal_matcher=__import__(
            "cerebros.jido.strategy", fromlist=["SignalMatcher"]
        ).SignalMatcher(patterns={SignalType.COMMAND}),
    )

    # Set up agent loop
    agent = AgentLoop(name="demo_agent")
    agent.register_strategy(strategy)

    # Process a signal
    signal = Signal(
        type=SignalType.COMMAND,
        data={"action": "test", "value": 42},
        source="user",
    )

    facts = agent.process_signal(signal)
    print(f"Signal: {signal}")
    print(f"Facts produced: {len(facts)}")
    for fact in facts:
        print(f"  Fact {fact.id}: {fact.value}")
    print(f"Agent signal log: {len(agent.signal_log)} signals")
    print(f"Agent fact store: {len(agent.fact_store)} facts")
    print()


def example_action_nodes():
    """Example 3: ActionNodes with schema introspection."""
    print("=" * 60)
    print("Example 3: ActionNodes with Schema Introspection")
    print("=" * 60)

    # Create NAS action nodes
    gen_node = GenerateArchitectureAction()
    build_node = BuildModelAction()
    train_node = TrainModelAction()
    eval_node = EvaluateModelAction()

    print(f"GenerateArchitecture schema: {gen_node.schema.params}")
    print(f"BuildModel schema: {build_node.schema.params}")
    print(f"TrainModel schema: {train_node.schema.params}")
    print(f"EvaluateModel schema: {eval_node.schema.params}")
    print()

    # Execute generate action directly
    context = {
        "minimum_levels": 3,
        "maximum_levels": 8,
        "minimum_units_per_level": 2,
        "maximum_units_per_level": 4,
        "minimum_neurons_per_unit": 8,
        "maximum_neurons_per_unit": 32,
        "seed": 42,
    }
    context = gen_node.run(context)
    arch = context.get("generate_architecture", {})
    print(f"Generated architecture:")
    print(f"  Levels: {arch.get('num_levels')}")
    print(f"  Total units: {arch.get('total_units')}")
    print(f"  Spec: {arch.get('spec')}")
    print(f"  Signals emitted: {len(gen_node.signals)}")
    for sig in gen_node.signals:
        print(f"    {sig}")
    print(f"  Facts produced: {len(gen_node.facts)}")
    print()


def example_nas_workflow():
    """Example 4: Full NAS workflow as a Runic DAG."""
    print("=" * 60)
    print("Example 4: NAS Workflow DAG")
    print("=" * 60)

    # Build NAS workflow using the builder
    builder = NASWorkflowBuilder(name="cifar10_nas")
    builder.with_architecture_params(
        minimum_levels=3,
        maximum_levels=12,
        minimum_units_per_level=2,
        maximum_units_per_level=6,
        minimum_neurons_per_unit=8,
        maximum_neurons_per_unit=128,
    )
    builder.with_training_params(
        epochs=10,
        batch_size=128,
        learning_rate=0.001,
    )
    builder.with_evaluation_metric("val_accuracy", direction="max")

    workflow = builder.build()

    # Visualize
    print(workflow.visualize())
    print()

    # Execute
    result = workflow.run({"seed": 42})
    print(f"Workflow status: {result.status.value}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Execution order: {result.execution_order}")
    print()

    # Show results from each stage
    for key in ["generate_architecture", "build_model", "train_model", "evaluate_model"]:
        val = result.context.get(key)
        if val:
            print(f"{key}:")
            if isinstance(val, dict):
                for k, v in val.items():
                    if k != "spec":
                        print(f"  {k}: {v}")
            print()


def example_nas_agent():
    """Example 5: Full NAS Agent with signal-driven search."""
    print("=" * 60)
    print("Example 5: NAS Agent - Signal-Driven Architecture Search")
    print("=" * 60)

    # Create a NAS agent
    agent = NASAgent.create(
        name="ames_regression_nas",
        minimum_levels=2,
        maximum_levels=8,
        minimum_units_per_level=1,
        maximum_units_per_level=4,
        minimum_neurons_per_unit=4,
        maximum_neurons_per_unit=32,
        epochs=7,
        batch_size=200,
        learning_rate=0.005,
        metric_to_rank_by="loss",
        direction="min",
    )

    print(f"Agent: {agent}")
    print(f"Workflow DAG:")
    print(agent.visualize_workflow())
    print()

    # Run search with multiple trials
    results = agent.run_search(num_trials=3, base_seed=42)

    print(f"\nSearch completed: {len(results)} trial results")
    print(f"Total signals processed: {len(agent.agent_loop.signal_log)}")
    print(f"Total facts produced: {len(agent.agent_loop.fact_store)}")
    print()

    # Show trial results
    for i, trial in enumerate(results):
        print(f"Trial {trial.get('trial_number', i)}:")
        for k, v in trial.items():
            if k not in ("spec", "all_values"):
                print(f"  {k}: {v}")
        print()

    # Introspection
    intro = Introspection(agent.agent_loop)
    print(f"Introspection stats: {intro.execution_stats()}")


def example_pipeline():
    """Example 6: Pipeline composition."""
    print("=" * 60)
    print("Example 6: Pipeline Composition")
    print("=" * 60)

    # Create a multi-step pipeline
    pipeline = Pipeline(
        name="data_pipeline",
        steps=[
            Step(name="normalize", func=lambda ctx: ctx.get("raw_value", 0) / 100.0),
            Step(name="scale", func=lambda ctx: ctx.get("normalize", 0) * 2.0),
            Step(name="offset", func=lambda ctx: ctx.get("scale", 0) + 0.5),
        ],
    )

    wf = Workflow(name="pipeline_demo")
    wf.add_component(pipeline)

    result = wf.run({"raw_value": 75})
    print(f"Input: 75")
    print(f"Normalized: {result.context.get('normalize')}")
    print(f"Scaled: {result.context.get('scale')}")
    print(f"Offset: {result.context.get('offset')}")
    print(f"Pipeline result: {result.context.get('data_pipeline')}")
    print()


if __name__ == "__main__":
    example_basic_workflow()
    example_signal_driven_agent()
    example_action_nodes()
    example_nas_workflow()
    example_nas_agent()
    example_pipeline()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
