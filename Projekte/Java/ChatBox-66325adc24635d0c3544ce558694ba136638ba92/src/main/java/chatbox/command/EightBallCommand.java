package chatbox.command;

import static chatbox.command.Command.random;
import static chatbox.command.Command.reply;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;

/**
 * Simulates a magic 8-ball.
 * 
 */
public class EightBallCommand implements Command {
	//@formatter:off
	private final String answers[] = {
		//positive
		"It is certain",
		"It is decidedly so",
		"Without a doubt",
		"Yes definitely",
		"You may rely on it",
		"As I see it, yes",
		"Most likely",
		"Outlook good",
		"Yes",
		"Signs point to yes",
		
		//neutral
		"Reply hazy try again",
		"Ask again later",
		"Better not tell you now",
		"Cannot predict now",
		"Concentrate and ask again",
		
		//negative
		"Don't count on it",
		"My reply is no",
		"My sources say no",
		"Outlook not so good",
		"Very doubtful",
		"That's impossible",
		"When pigs fly",
		"Chances are lower than skynet",
		"Are you kidding? No!",
		"No way"
	};
	//@formatter:on

	@Override
	public String name() {
		return "8ball";
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Simulates a magic 8-ball.")
			.example("Is Java the best?", "")
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		String answer = random(answers);
		return reply(answer, chatCommand);
	}
}
