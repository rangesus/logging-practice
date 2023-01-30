package chatbox.command;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;
import chatbox.filter.GrootFilter;

/**
 * Makes the bot talk in Groot.
 * 
 */
public class GrootCommand implements Command {
	private final GrootFilter filter;

	public GrootCommand(GrootFilter filter) {
		this.filter = filter;
	}

	@Override
	public String name() {
		return "groot";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("I am Groot.")
			.detail("Toggles a filter that makes Oak speak in Groot.")
			.includeSummaryWithDetail(false)
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		int roomId = chatCommand.getMessage().getRoomId();
		filter.toggle(roomId);
		return null;
	}
}
