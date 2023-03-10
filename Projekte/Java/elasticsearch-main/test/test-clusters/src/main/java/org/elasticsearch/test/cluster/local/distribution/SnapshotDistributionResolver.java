/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.test.cluster.local.distribution;

import org.elasticsearch.test.cluster.util.Version;

public class SnapshotDistributionResolver implements DistributionResolver {
    @Override
    public DistributionDescriptor resolve(Version version, DistributionType type) {
        // Not yet implemented
        throw new UnsupportedOperationException("Cannot resolve distribution for version " + version);
    }
}
