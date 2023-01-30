package chatbox.bot;

import static org.mockito.Mockito.mock;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import chatbox.Database;
import chatbox.command.AboutCommand;
import chatbox.command.AdventOfCodeCommand;
import chatbox.command.AfkCommand;
import chatbox.command.CatCommand;
import chatbox.command.Command;
import chatbox.command.EightBallCommand;
import chatbox.command.FacepalmCommand;
import chatbox.command.FatCatCommand;
import chatbox.command.GrootCommand;
import chatbox.command.HelpCommand;
import chatbox.command.HelpDoc;
import chatbox.command.ReactCommand;
import chatbox.command.RollCommand;
import chatbox.command.RolloverCommand;
import chatbox.command.ShrugCommand;
import chatbox.command.ShutdownCommand;
import chatbox.command.SummonCommand;
import chatbox.command.TagCommand;
import chatbox.command.UnsummonCommand;
import chatbox.command.WaduCommand;
import chatbox.command.WikiCommand;
import chatbox.command.define.DefineCommand;
import chatbox.command.effective.EffectiveJavaCommand;
import chatbox.command.http.HttpCommand;
import chatbox.command.javadoc.JavadocCommand;
import chatbox.command.learn.LearnCommand;
import chatbox.command.learn.LearnedCommandsDao;
import chatbox.command.learn.UnlearnCommand;
import chatbox.command.urban.UrbanCommand;
import chatbox.listener.Listener;
import chatbox.listener.MentionListener;
import chatbox.listener.MornListener;
import chatbox.listener.WaveListener;
import chatbox.listener.WelcomeListener;
import chatbox.util.ChatBuilder;


public class CommandsWikiPage {
	public static void main(String args[]) {
		Database db = mock(Database.class);
		String trigger = "/";
		LearnedCommandsDao learnedCommands = new LearnedCommandsDao();

		List<Listener> listeners = new ArrayList<>();
		{
			MentionListener mentionListener = new MentionListener("");

			listeners.add(mentionListener);
			listeners.add(new MornListener("ChatBox", 1000, mentionListener));
			listeners.add(new WaveListener("ChatBox", 1000, mentionListener));
			listeners.add(new WelcomeListener(db, Collections.emptyMap()));

			listeners.removeIf(l -> l.name() == null);
			listeners.sort((a, b) -> a.name().compareTo(b.name()));
		}

		List<Command> commands = new ArrayList<>();
		{
			commands.add(new AboutCommand(null, null));
			commands.add(new AdventOfCodeCommand(Collections.emptyMap(), null));
			commands.add(new AfkCommand());
			commands.add(new CatCommand(null));
			commands.add(new DefineCommand(null));
			commands.add(new EffectiveJavaCommand());
			commands.add(new EightBallCommand());
			commands.add(new FacepalmCommand(""));
			commands.add(new FatCatCommand(db));
			commands.add(new GrootCommand(null));
			commands.add(new HelpCommand(commands, learnedCommands, listeners));
			commands.add(new HttpCommand());
			commands.add(new JavadocCommand(null));
			commands.add(new LearnCommand(commands, learnedCommands));
			commands.add(new ReactCommand(null));
			commands.add(new RollCommand());
			commands.add(new RolloverCommand(null));
			commands.add(new ShrugCommand());
			commands.add(new ShutdownCommand());
			commands.add(new SummonCommand(2));
			commands.add(new TagCommand());
			commands.add(new UnlearnCommand(commands, learnedCommands));
			commands.add(new UnsummonCommand());
			commands.add(new UrbanCommand());
			commands.add(new WaduCommand(null));
			commands.add(new WikiCommand());

			commands.sort((a, b) -> a.name().compareTo(b.name()));
		}

		ChatBuilder cb = new ChatBuilder();

		cb.append("This page lists all of ChatBox commands and listeners.");
		cb.nl().nl().append("Type ").code("/help COMMAND").append(" to see this help documentation in chat.");

		cb.nl().nl().append("# Commands");
		if (commands.isEmpty()) {
			cb.nl().nl().italic("no commands defined");
		} else {
			for (Command command : commands) {
				cb.nl().nl().append("## ").append(trigger).append(command.name()).nl().nl();

				HelpDoc help = command.help();
				if (help.isIncludeSummaryWithDetail()) {
					cb.append(help.getSummary());
				}
				if (help.getDetail() != null) {
					cb.append(" ").append(help.getDetail());
				}

				List<String[]> examples = help.getExamples();
				if (!examples.isEmpty()) {
					cb.nl().nl().bold("Examples:").nl();

					for (String[] example : examples) {
						String parameters = example[0];
						String description = example[1];

						cb.nl().append(" * ").code().append(trigger).append(command.name());
						if (!parameters.isEmpty()) {
							cb.append(" ").append(parameters);
						}
						cb.code();
						if (!description.isEmpty()) {
							cb.append(" - ").append(description);
						}
					}
				}

				Collection<String> aliases = command.aliases();
				if (!aliases.isEmpty()) {
					cb.nl().nl().bold("Aliases:").append(" ");

					boolean first = true;
					for (String alias : aliases) {
						if (first) {
							first = false;
						} else {
							cb.append(", ");
						}

						cb.code(alias);
					}
				}
			}
		}

		cb.nl().nl().append("# Listeners");
		if (listeners.isEmpty()) {
			cb.nl().nl().italic("no listeners defined");
		} else {
			for (Listener listener : listeners) {
				HelpDoc help = listener.help();

				cb.nl().nl().append("## ").append(listener.name()).nl().nl();
				if (help.isIncludeSummaryWithDetail()) {
					cb.append(help.getSummary());
				}
				if (help.getDetail() != null) {
					cb.append(" ").append(help.getDetail());
				}
			}
		}

		System.out.println(cb);
	}
}
