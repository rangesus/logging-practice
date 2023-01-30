package chatbox.command;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;
import chatbox.filter.UpsidedownTextFilter;

/**
 * Turns the bot upside down.
 * 
 */
public class RolloverCommand implements Command {
	private final UpsidedownTextFilter filter;

	public RolloverCommand(UpsidedownTextFilter filter) {
		this.filter = filter;
	}

	@Override
	public String name() {
		return "rollover";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Turns the bot upside down.")
			.detail("Toggles a filter that makes all the letters in the messages Oak posts look like they are upside down.")
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
