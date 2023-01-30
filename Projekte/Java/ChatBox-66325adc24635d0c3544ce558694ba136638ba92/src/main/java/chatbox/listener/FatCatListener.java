package chatbox.listener;

import chatbox.bot.BotContext;
import chatbox.bot.ChatResponse;
import chatbox.chat.ChatMessage;
import chatbox.command.FatCatCommand;


public class FatCatListener implements Listener {
	private final FatCatCommand command;

	public FatCatListener(FatCatCommand command) {
		this.command = command;
	}

	@Override
	public ChatResponse onMessage(ChatMessage message, BotContext context) {
		String reply = command.handleResponse(message);
		return (reply == null) ? null : Listener.reply(reply, message);
	}
}
