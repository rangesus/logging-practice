package chatbox.bot;

import chatbox.command.Command;

/**
 * Handles a unrecognized command.
 * 
 */
public interface UnknownCommandHandler extends Command {
	default String name() {
		return null;
	}

	default String description() {
		return null;
	}

	default String helpText(String trigger) {
		return null;
	}
}
