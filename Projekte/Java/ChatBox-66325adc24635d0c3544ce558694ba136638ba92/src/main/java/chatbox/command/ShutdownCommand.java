package chatbox.command;

import static chatbox.command.Command.reply;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;

/**
 * Shuts down the bot.
 * 
 */
public class ShutdownCommand implements Command {

	@Override
	public String name() {
		return "shutdown";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Terminates the bot (admins only).")
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		if (context.isAuthorAdmin()) {
			boolean broadcast = chatCommand.getContent().equals("broadcast");
			context.shutdownBot("Shutting down. See you later.", broadcast);
			return null;
		}

		return reply("Only admins can shut me down. :P", chatCommand);
	}
}
