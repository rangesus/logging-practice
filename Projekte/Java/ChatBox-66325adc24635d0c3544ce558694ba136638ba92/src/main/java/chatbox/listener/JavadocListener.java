package chatbox.listener;

import chatbox.bot.BotContext;
import chatbox.bot.ChatResponse;
import chatbox.chat.ChatMessage;
import chatbox.command.javadoc.JavadocCommand;


public class JavadocListener implements Listener {
	private final JavadocCommand command;

	public JavadocListener(JavadocCommand command) {
		this.command = command;
	}

	@Override
	public ChatResponse onMessage(ChatMessage message, BotContext context) {
		String content = message.getContent().getContent();
		try {
			int num = Integer.parseInt(content);
			return command.showChoice(message, num);
		} catch (NumberFormatException e) {
			return null;
		}
	}
}
