package chatbox.command;

import static chatbox.command.Command.reply;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.http.HttpStatus;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.xml.sax.SAXException;

import com.google.common.net.UrlEscapers;

import chatbox.bot.BotContext;
import chatbox.bot.ChatCommand;
import chatbox.bot.ChatResponse;
import chatbox.chat.SplitStrategy;
import chatbox.util.ChatBuilder;
import chatbox.util.Leaf;

/**
 * Displays a random cat picture.
 *
 */
public class CatCommand implements Command {
	private static final Logger logger = Logger.getLogger(CatCommand.class.getName());

	private final String requestUrl;

	public CatCommand(String key) {
		String url = "http://thecatapi.com/api/images/get?size=small&format=xml&type=gif";
		if (key != null) {
			url += "&api_key=" + UrlEscapers.urlFormParameterEscaper().escape(key);
		}
		requestUrl = url;
	}

	@Override
	public String name() {
		return "cat";
	}

	@Override
	public List<String> aliases() {
		return Arrays.asList("meow");
	}

	@Override
	public HelpDoc help() {
		//@formatter:off
		return new HelpDoc.Builder(this)
			.summary("Displays a random cat picture. :3")
			.detail("Images from thecatapi.com.")
		.build();
		//@formatter:on
	}

	@Override
	public ChatResponse onMessage(ChatCommand chatCommand, BotContext context) {
		int repeats = 0;
		try (CloseableHttpClient client = createClient()) {
			while (repeats < 5) {
				String catUrl = nextCat(client);
				if (isCatThere(client, catUrl)) {
					return new ChatResponse(catUrl, SplitStrategy.NONE, true);
				}

				repeats++;
			}
		} catch (IOException | SAXException e) {
			logger.log(Level.SEVERE, "Problem getting cat.", e);

			//@formatter:off
			return new ChatResponse(new ChatBuilder()
				.reply(chatCommand)
				.append("Error getting cat: ")
				.code(e.getMessage())
			);
			//@formatter:on
		}

		return reply("No cats found. Try again. :(", chatCommand);
	}

	/**
	 * Gets a random cat picture.
	 * @param client the HTTP client
	 * @return the URL to the cat picture
	 * @throws IOException if there's a network error
	 * @throws SAXException if there's a problem parsing the XML response
	 */
	private String nextCat(CloseableHttpClient client) throws IOException, SAXException {
		HttpGet request = new HttpGet(requestUrl);
		try (CloseableHttpResponse response = client.execute(request)) {
			Leaf document;
			try (InputStream in = response.getEntity().getContent()) {
				document = Leaf.parse(in);
			}

			Leaf urlElement = document.selectFirst("/response/data/images/image/url");
			return urlElement.text();
		}
	}

	/**
	 * Checks to see if the given image exists. Some of the URLs that the API
	 * returns don't work anymore.
	 * @param client the HTTP client
	 * @param url the URL to the image
	 * @return true if the image exists, false if not
	 */
	private boolean isCatThere(CloseableHttpClient client, String url) {
		HttpHead request = new HttpHead(url);
		try (CloseableHttpResponse response = client.execute(request)) {
			return response.getStatusLine().getStatusCode() == HttpStatus.SC_OK;
		} catch (IOException e) {
			return false;
		}
	}

	/**
	 * Creates an HTTP client. This method is for unit testing.
	 * @return the HTTP client
	 */
	CloseableHttpClient createClient() {
		return HttpClients.createDefault();
	}
}
