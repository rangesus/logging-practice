package chatbox.command;

import static chatbox.command.Command.reply;

import com.google.common.net.UrlEscapers;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;

/**
 * Displays on-boxed Wikipedia pages.
 * 
 */
public class WikiCommand implements Command {
	@Override
	public String name() {
		return "wiki";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Displays a one-box for a Wikipedia page.")
			.example("James Gosling", "Displays a one-box for the \"James Gosling\" Wikipedia page.")
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		String content = chatCommand.getContent().trim();
		if (content.isEmpty()) {
			return reply("Please specify the term you'd like to display.", chatCommand);
		}

		String url = "http://en.wikipedia.org/wiki/" + UrlEscapers.urlPathSegmentEscaper().escape(content);
		return new ChatResponse(url);
	}
}
