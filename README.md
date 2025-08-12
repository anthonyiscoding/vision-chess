# Caveats / Limitations

This model doesn't enforce a maximum context length because the current token encoding scheme depends on all previous moves for context / meaning. This also means that currently the model will only be able to play standard chess and not variants like Chess960.

The current encoding scheme has been chosen solely because it is simple to implement.