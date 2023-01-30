package chatbox.command;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

import chatbox.Main;
import chatbox.Statistics;
import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;
import chatbox.chat.SplitStrategy;
import chatbox.util.ChatBuilder;
import chatbox.util.RelativeDateFormat;

/**
 * Displays meta information about the bot.
 * 
 */
public class AboutCommand implements Command {
	private final LocalDateTime startedUp = LocalDateTime.now();
	private final Statistics stats;
	private final String host;

	public AboutCommand(Statistics stats, String host) {
		this.stats = stats;
		this.host = host;
	}

	@Override
	public String name() {
		return "about";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Displays information about this bot.")
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		RelativeDateFormat relativeDf = new RelativeDateFormat();
		LocalDateTime built = LocalDateTime.ofInstant(Main.BUILT, ZoneId.systemDefault());

		//@formatter:off
		ChatBuilder cb = new ChatBuilder()
		.bold("Chatbox").append(" by ").link("Murali", "").append(" | ")
		.link("source code", Main.URL).append(" | ")
		.append("JAR built on: ").append(relativeDf.format(built)).append(" | ")
		.append("started up: ").append(relativeDf.format(startedUp));
		//@formatter:on

		if (host != null) {
			cb.append(" | ").append("hosted by: ").append(host);
		}

		if (stats != null) {
			cb.append(" | ").append("responded to ").append(stats.getMessagesRespondedTo()).append(" commands");

			LocalDateTime since = stats.getSince();
			if (since != null) {
				DateTimeFormatter formatter = DateTimeFormatter.ofPattern("MMM d, yyyy");
				cb.append(" since ").append(since.format(formatter));
			}
		}

		return new ChatResponse(cb, SplitStrategy.WORD);
	}
}
