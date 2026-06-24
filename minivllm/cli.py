"""Command-line entrypoint for mini-vLLM."""


def main() -> None:
    """Run the CLI.

    The real generation command is added after the model runner and engine
    exist. Keeping the entrypoint now makes packaging and smoke tests concrete.
    """
    print("mini-vLLM is initialized. Engine implementation is coming next.")


if __name__ == "__main__":
    main()
