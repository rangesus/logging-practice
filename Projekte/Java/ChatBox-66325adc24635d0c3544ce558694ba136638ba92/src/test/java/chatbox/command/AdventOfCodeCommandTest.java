package chatbox.command;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;
import chatbox.command.AdventOfCodeApi;
import chatbox.command.AdventOfCodeCommand;
import chatbox.util.ChatCommandBuilder;

/**
 * 
 */
public class AdventOfCodeCommandTest {
	private final ChatCommandBuilder chatCommandBuilder = new ChatCommandBuilder(new AdventOfCodeCommand(new HashMap<>(), null).name());

	@Test
	public void using_default_id() {
		ChatCommand message = chatCommandBuilder.build(1, 1, "");

		Map<Integer, String> leaderboardIds = new HashMap<>();
		leaderboardIds.put(1, "123456");
		AdventOfCodeCommand command = mock(leaderboardIds, "123456");

		ChatResponse response = command.onMessage(message, mockBotContext());
		
	}

	@Test
	public void no_default_id() {
		ChatCommand message = chatCommandBuilder.build(1, 1, "");

		Map<Integer, String> leaderboardIds = new HashMap<>();
		AdventOfCodeApi api = new AdventOfCodeApi("") {
			@Override
			JsonNode get(String url) throws IOException {
				fail("Should not be called because no leaderboard ID was specified.");
				return null;
			}
		};
		AdventOfCodeCommand command = new AdventOfCodeCommand(leaderboardIds, api) {
			@Override
			boolean isActive() {
				return true;
			}
		};

		BotContext context = new BotContext(false, "/", null, Collections.emptyList(), Collections.emptyList(), null);
		ChatResponse response = command.onMessage(message, context);
		assertEquals(":1 Please specify a leaderboard ID (e.g. /advent 123456).", response.getMessage());
	}

	@Test
	public void override_default_id() {
		ChatCommand message = chatCommandBuilder.build(1, 1, "098765");

		Map<Integer, String> leaderboardIds = new HashMap<>();
		leaderboardIds.put(1, "123456");
		AdventOfCodeCommand command = mock(leaderboardIds, "098765");

		ChatResponse response = command.onMessage(message, mockBotContext());
		
	}

	@Test
	public void not_active() {
		ChatCommand message = chatCommandBuilder.build(1, 1, "");

		Map<Integer, String> leaderboardIds = new HashMap<>();
		AdventOfCodeApi api = new AdventOfCodeApi("") {
			@Override
			JsonNode get(String url) throws IOException {
				fail("Should not be called because the command is not active.");
				return null;
			}
		};
		AdventOfCodeCommand command = new AdventOfCodeCommand(leaderboardIds, api) {
			@Override
			boolean isActive() {
				return false;
			}
		};

		BotContext context = new BotContext(false, "/", null, Collections.emptyList(), Collections.emptyList(), null);
		ChatResponse response = command.onMessage(message, context);
		assertEquals(":1 This command is only active during the month of December.", response.getMessage());
	}

	private static AdventOfCodeCommand mock(Map<Integer, String> leaderboardIds, String expectedLeaderboardId) {
		AdventOfCodeApi api = new AdventOfCodeApi("") {
			@Override
			JsonNode get(String url) throws IOException {
				int year = LocalDateTime.now().getYear();
				assertEquals("http://adventofcode.com/" + year + "/leaderboard/private/view/" + expectedLeaderboardId + ".json", url);

				ObjectMapper mapper = new ObjectMapper();
				try (InputStream in = getClass().getResourceAsStream("advent-of-code-2017.json")) {
					return mapper.readTree(in);
				}
			}
		};
		return new AdventOfCodeCommand(leaderboardIds, api) {
			@Override
			boolean isActive() {
				return true;
			}
		};
	}

	private static BotContext mockBotContext() {
		return new BotContext(false, "/", null, Collections.emptyList(), Collections.emptyList(), 1);
	}
}
