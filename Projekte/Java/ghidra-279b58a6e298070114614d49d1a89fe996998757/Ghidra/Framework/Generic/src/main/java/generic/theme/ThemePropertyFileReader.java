/* ###
 * IP: GHIDRA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package generic.theme;

import java.io.*;

import generic.jar.ResourceFile;

/**
 * Reads the values for a single theme.properities file
 */
public class ThemePropertyFileReader extends AbstractThemeReader {

	private GThemeValueMap defaults;
	private GThemeValueMap darkDefaults;

	/**
	 * Constructor for when the the theme.properties file is a {@link ResourceFile}
	 * @param file the {@link ResourceFile} esourceFileto read
	 * @throws IOException if an I/O error occurs reading the file
	 */
	public ThemePropertyFileReader(ResourceFile file) throws IOException {
		super(file.getAbsolutePath());

		try (Reader reader = new InputStreamReader(file.getInputStream())) {
			read(reader);
		}

	}

	/**
	 * Constructor using a Reader (needed for reading from zip files).
	 * @param source the name or description of the Reader source
	 * @param reader the {@link Reader} to parse as theme data
	 * @throws IOException if an I/O error occurs while reading from the Reader
	 */
	protected ThemePropertyFileReader(String source, Reader reader) throws IOException {
		super(source);
		read(reader);
	}

	/**
	 * Returns the map of standard defaults values. 
	 * @return the map of standard defaults values.
	 */
	public GThemeValueMap getDefaultValues() {
		return defaults == null ? new GThemeValueMap() : defaults;
	}

	/**
	 * Returns the map of dark defaults values. 
	 * @return the map of dark defaults values.
	 */
	public GThemeValueMap getDarkDefaultValues() {
		return darkDefaults == null ? new GThemeValueMap() : darkDefaults;
	}

	protected void processNoSection(Section section) throws IOException {
		if (!section.isEmpty()) {
			error(0, "Theme properties file has values defined outside of a defined section");
		}
	}

	@Override
	protected void processDefaultSection(Section section) throws IOException {
		defaults = new GThemeValueMap();
		processValues(defaults, section);
	}

	@Override
	protected void processDarkDefaultSection(Section section) throws IOException {
		darkDefaults = new GThemeValueMap();
		processValues(darkDefaults, section);
	}

}
