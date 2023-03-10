package cn.hutool.extra.compress.extractor;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.IORuntimeException;
import cn.hutool.core.io.IoUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.lang.Filter;
import cn.hutool.core.util.StrUtil;
import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;
import org.apache.commons.compress.utils.SeekableInMemoryByteChannel;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.SeekableByteChannel;
import java.util.RandomAccess;

/**
 * 7z格式数据解压器，即将归档打包的数据释放
 *
 * @author looly
 * @since 5.5.0
 */
public class SevenZExtractor implements Extractor, RandomAccess {

	private final SevenZFile sevenZFile;

	/**
	 * 构造
	 *
	 * @param file 包文件
	 */
	public SevenZExtractor(File file) {
		this(file, null);
	}

	/**
	 * 构造
	 *
	 * @param file     包文件
	 * @param password 密码，null表示无密码
	 */
	public SevenZExtractor(File file, char[] password) {
		try {
			this.sevenZFile = new SevenZFile(file, password);
		} catch (IOException e) {
			throw new IORuntimeException(e);
		}
	}

	/**
	 * 构造
	 *
	 * @param in 包流
	 */
	public SevenZExtractor(InputStream in) {
		this(in, null);
	}

	/**
	 * 构造
	 *
	 * @param in       包流
	 * @param password 密码，null表示无密码
	 */
	public SevenZExtractor(InputStream in, char[] password) {
		this(new SeekableInMemoryByteChannel(IoUtil.readBytes(in)), password);
	}

	/**
	 * 构造
	 *
	 * @param channel {@link SeekableByteChannel}
	 */
	public SevenZExtractor(SeekableByteChannel channel) {
		this(channel, null);
	}

	/**
	 * 构造
	 *
	 * @param channel  {@link SeekableByteChannel}
	 * @param password 密码，null表示无密码
	 */
	public SevenZExtractor(SeekableByteChannel channel, char[] password) {
		try {
			this.sevenZFile = new SevenZFile(channel, password);
		} catch (IOException e) {
			throw new IORuntimeException(e);
		}
	}

	/**
	 * 释放（解压）到指定目录，结束后自动关闭流，此方法只能调用一次
	 *
	 * @param targetDir 目标目录
	 * @param filter    解压文件过滤器，用于指定需要释放的文件，null表示不过滤。当{@link Filter#accept(Object)}为true时释放。
	 */
	@Override
	public void extract(File targetDir, int stripComponents, Filter<ArchiveEntry> filter) {
		try {
			extractInternal(targetDir, stripComponents, filter);
		} catch (IOException e) {
			throw new IORuntimeException(e);
		} finally {
			close();
		}
	}

	/**
	 * 获取满足指定过滤要求的压缩包内的第一个文件流
	 *
	 * @param filter 用于指定需要释放的文件，null表示不过滤。当{@link Filter#accept(Object)}为true时返回对应流。
	 * @return 满足过滤要求的第一个文件的流, 无满足条件的文件返回{@code null}
	 * @since 5.7.14
	 */
	public InputStream getFirst(Filter<ArchiveEntry> filter) {
		final SevenZFile sevenZFile = this.sevenZFile;
		for (SevenZArchiveEntry entry : sevenZFile.getEntries()) {
			if (null != filter && false == filter.accept(entry)) {
				continue;
			}
			if (entry.isDirectory()) {
				continue;
			}

			try {
				return sevenZFile.getInputStream(entry);
			} catch (IOException e) {
				throw new IORuntimeException(e);
			}
		}

		return null;
	}

	/**
	 * 获取指定名称的文件流
	 *
	 * @param entryName entry名称
	 * @return 文件流，无文件返回{@code null}
	 * @since 5.7.14
	 */
	public InputStream get(String entryName) {
		return getFirst((entry) -> StrUtil.equals(entryName, entry.getName()));
	}

	/**
	 * 释放（解压）到指定目录
	 *
	 * @param targetDir       目标目录
	 * @param stripComponents 清除(剥离)压缩包里面的 n 级文件夹名
	 * @param filter          解压文件过滤器，用于指定需要释放的文件，null表示不过滤。当{@link Filter#accept(Object)}为true时释放。
	 * @throws IOException IO异常
	 */
	private void extractInternal(File targetDir, int stripComponents, Filter<ArchiveEntry> filter) throws IOException {
		Assert.isTrue(null != targetDir && ((false == targetDir.exists()) || targetDir.isDirectory()), "target must be dir.");
		final SevenZFile sevenZFile = this.sevenZFile;
		SevenZArchiveEntry entry;
		File outItemFile;
		while (null != (entry = this.sevenZFile.getNextEntry())) {
			if (null != filter && false == filter.accept(entry)) {
				continue;
			}
			String entryName = this.stripName(entry.getName(), stripComponents);
			if (entryName == null) {
				// 剥离文件夹层级
				continue;
			}
			outItemFile = FileUtil.file(targetDir, entryName);
			if (entry.isDirectory()) {
				// 创建对应目录
				//noinspection ResultOfMethodCallIgnored
				outItemFile.mkdirs();
			} else if (entry.hasStream()) {
				// 读取entry对应数据流
				FileUtil.writeFromStream(new Seven7EntryInputStream(sevenZFile, entry), outItemFile);
			} else {
				// 无数据流的文件创建为空文件
				FileUtil.touch(outItemFile);
			}
		}
	}

	@Override
	public void close() {
		IoUtil.close(this.sevenZFile);
	}
}
