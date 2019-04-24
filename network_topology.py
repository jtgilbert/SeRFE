""" Adds attributes to a line network to give it topology (i.e. provide a way to determine which reaches are
upstream/downstream of any other given reaches. A reach ID is generated ('rid') with form <number1.number2>,
where number1 is a 'chain' and number two is a 'reach' within that chain.  The tool identifies the start of a
chain (headwater reach) and works its way through it reach by reach.  When the end of a chain is reached, the
tool looks for a new chain to start until there are no remaining chains to attribute.  Additional attributes
 identify the next reach downstream 'rid_ds' and upstream 'ris_us' of a given reach, as well as an attribute
 to identify confluence reaches. """

# imports
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from rasterstats import zonal_stats


class NetworkTopology:

    def __init__(self, network, dem):
        """
        Adds fields representing network topology to each segment of an input network.
        Fields:
        'rid': a reach ID of form <chain>.<reach> where chain and reach are each integers
        'confluence': binary, 0 indicates the reach is not a confluence, 1 indicates that it is a confluence
        'rid_ds': the 'rid' of the next reach downstream
        'rid_us': the 'rid' of the next reach upstream
        'rid_us2': if the given segment is a confluence, this is the 'rid' of the other upstream segment
        :param network: string - path to a drainage network shapefile
        :param dem: string - path to a Digital Elevation Model
        """

        self.network = network
        stream_network = gpd.read_file(self.network)

        # get all start and end coordinates and start elevations
        iden = []
        start_coord = []
        end_coord = []
        start_el = []

        for i in stream_network.index:
            reach_geom = stream_network.loc[i, 'geometry']

            start_x_coord = round(reach_geom.boundary[0].xy[0][0], 2)
            start_y_coord = round(reach_geom.boundary[0].xy[1][0], 2)

            end_x_coord = round(reach_geom.boundary[1].xy[0][0], 2)
            end_y_coord = round(reach_geom.boundary[1].xy[1][0], 2)

            start_coord.append((start_x_coord, start_y_coord))
            end_coord.append((end_x_coord, end_y_coord))

            pt = Point(start_x_coord, start_y_coord)
            buf = pt.buffer(20)
            zs = zonal_stats(buf, dem, stats='max')
            elev = zs[0].get('max')

            start_el.append(elev)
            iden.append(i)

        # set up table (dataframe) to work with
        d = {'ID': iden, 'start_coord': start_coord, 'end_coord': end_coord, 'elev': start_el, 'rid': 0, 'rid_us': 0,
             'rid_ds': 0, 'confluence': 0, 'rid_us2': -9999}

        self.df = pd.DataFrame(data=d)
        self.find_network_rids(df=self.df)
        self.find_ds_rids(self.df)
        self.find_confluences(self.df)
        self.find_us_rids(self.df)
        self.add_topology_fields(self.df, stream_network, self.network)

    def find_chain_start(self, df, chain):
        """ finds the start of the next chain in network sequence """

        sub_df = df[df['rid'] == 0]
        if len(sub_df) > 0:
            for i in sub_df.index:
                if df.loc[i, 'elev'] == sub_df['elev'].max():
                    seg_id = sub_df.loc[i, 'ID']
                    df.loc[seg_id, 'rid'] = '{}.1'.format(chain)

                    return int(seg_id)

        else:
            return None

    def find_next_in_chain(self, df, prev_id, chain, reach):
        """ finds the next network segment in a chain """

        for i in df.index:
            if df.loc[i, 'start_coord'] == df.loc[prev_id, 'end_coord']:
                seg_id = df.loc[i, 'ID']
                if df.loc[seg_id, 'rid'] == 0:
                    df.loc[seg_id, 'rid'] = '{0}.{1}'.format(chain, reach)

                    return int(seg_id)

                else:
                    return None

    def find_network_rids(self, df):
        """ attributes segment with reach id 'rid' """

        chain = 1

        start = self.find_chain_start(df=df, chain=chain)

        while start is not None:

            reach = 2
            new_seg = self.find_next_in_chain(df=df, prev_id=start, chain=chain, reach=reach)
            prev_seg = new_seg
            reach = 3

            while prev_seg is not None:
                new_seg = self.find_next_in_chain(df=df, prev_id=prev_seg, chain=chain, reach=reach)
                prev_seg = new_seg
                reach += 1

            chain += 1

            start = self.find_chain_start(df=df, chain=chain)

        return

    def find_ds_rids(self, df):
        """ finds the downstream reach id 'rid_ds' for segment """

        for i in df.index:
            for j in df.index:
                if df.loc[j, 'start_coord'] == df.loc[i, 'end_coord']:
                    df.loc[i, 'rid_ds'] = df.loc[j, 'rid']

        return

    def find_confluences(self, df):
        """ finds and attributes the confluences within the drainage network """

        confluence_ids = []
        for i in df.index:
            for j in df.index:
                if (df.loc[i, 'end_coord'] == df.loc[j, 'end_coord']) & (df.loc[i, 'ID'] != df.loc[j, 'ID']):
                    confluence_ids.append(df.loc[i, 'rid'])

        confluence_coords = []
        #for x in range(len(confluence_ids)):
        for i in df.index:
            if df.loc[i, 'rid'] in confluence_ids:
                confluence_coords.append(df.loc[i, 'end_coord'])

        #for x in range(len(confluence_coords)):
        for i in df.index:
            if df.loc[i, 'start_coord'] in confluence_coords:
                df.loc[i, 'confluence'] = 1

        return

    def find_us_rids(self, df):
        """ finds the upstream reach ids 'rid_us', 'rid_us2' for segment """

        for i in df.index:
            for j in df.index:
                if df.loc[j, 'end_coord'] == df.loc[i, 'start_coord']:
                    df.loc[i, 'rid_us'] = df.loc[j, 'rid']

        con_ids = []
        for i in df.index:
            if df.loc[i, 'confluence'] == 1:
                con_ids.append(df.loc[i, 'ID'])

        for x in range(len(con_ids)):
            con_reaches = []
            s_c = df.loc[con_ids[x], 'start_coord']
            for i in df.index:
                if df.loc[i, 'end_coord'] == s_c:
                    con_reaches.append(df.loc[i, 'rid'])

            if len(con_reaches) > 1:

                if df.loc[con_ids[x], 'rid_us'] == con_reaches[0]:
                    df.loc[con_ids[x], 'rid_us2'] = con_reaches[1]
                elif df.loc[con_ids[x], 'rid_us'] == con_reaches[1]:
                    df.loc[con_ids[x], 'rid_us2'] = con_reaches[0]

        return

    def add_topology_fields(self, df, network, output):
        """ adds all of the new attribute information to the input drainage network shapefile """

        network['confluence'] = df['confluence']
        network['rid'] = df['rid']
        network['rid_ds'] = df['rid_ds']
        network['rid_us'] = df['rid_us']
        network['rid_us2'] = df['rid_us2']
        network.to_file(output)

        return


# Useful functions for dealing with topology contained in a class

class TopologyTools:
    """
    This class provides methods to deal with topology. An instance is created using a stream network shapefile,
    after which methods can be used that provide topological functions to a network that has already been attributed
    with the topology fields created using the 'Network Topology' class
    """

    def __init__(self, network):
        """

        :param network: string - path to a stream network shapefile
        """

        self.networkpath = network
        self.network = gpd.read_file(network)

    def find_us_seg(self, segid):
        """
        Finds the next segment upstream of a selected segment
        :param segid: int - identifier (not 'rid')
        :return: int - identifier of next upstream segment
        """

        if self.network.loc[segid, 'rid_us'] == '0':
            return None

        else:
            for i in self.network.index:
                if self.network.loc[i, 'rid'] == self.network.loc[segid, 'rid_us']:
                    usid = i

            return usid

    def find_us_seg2(self, segid):
        """
        Find the second upstream segment of a selected segment if selected segment is a confluence
        :param segid: int - identifier (not 'rid')
        :return: int - identifier of second upstream segment (at confluence)
        """

        if self.network.loc[segid, 'rid_us2'] == '-9999':
            return None

        else:
            for i in self.network.index:
                if self.network.loc[i, 'rid'] == self.network.loc[segid, 'rid_us2']:
                    usid = i

            return usid

    def find_all_us(self, segid):
        """
        Finds all segments upstream of a a given segment
        :param segid: int - indentifier (not 'rid')
        :return: list - contains identifiers of all upstream segments
        """

        us_list = []
        branch_list = []

        if self.network.loc[segid, 'rid_us2'] != '-9999':
            branch_list.append(segid)

        usid = self.find_us_seg(segid)
        if usid is not None:
            us_list.append(usid)
            if self.network.loc[usid, 'rid_us2'] != '-9999':
                branch_list.append(usid)

        while usid is not None:
            segmentid = usid
            usid = self.find_us_seg(segmentid)
            if usid is not None:
                if self.network.loc[usid, 'rid_us2'] != '-9999':
                    branch_list.append(usid)
                us_list.append(usid)

        while len(branch_list) > 0:

            for segment in branch_list:

                usid2 = self.find_us_seg2(segment)

                if usid2 is not None:
                    us_list.append(usid2)
                    if self.network.loc[usid2, 'rid_us2'] != '-9999':
                        branch_list.append(usid2)

                while usid2 is not None:
                    segmentid2 = usid2
                    usid2 = self.find_us_seg(segmentid2)
                    if usid2 is not None:
                        if self.network.loc[usid2, 'rid_us2'] != '-9999':
                            branch_list.append(usid2)
                        us_list.append(usid2)

                branch_list.remove(segment)

        return us_list

    def find_all_ds(self, segid):
        """
        Finds all segments downstream of a given segment
        :param segid: int - identifier (not 'rid')
        :return: list - contains identifiers of all downwstream segments
        """

        ds_list = []

        dsid = self.network.loc[segid, 'rid_ds']

        while dsid != '0':

            for i in self.network.index:
                if self.network.loc[i, 'rid'] == dsid:
                    ds_list.append(i)
                    dsid = self.network.loc[i, 'rid_ds']

        return ds_list

    def get_next_reach(self, current_segment):
        """
        Find the next reach in a chain downstream of a chosen segment
        :param current_segment: int - identifier of chosen segment
        :return: int - identifier of next reach in chain
        """

        rids = []
        for i in self.network.index:
            rid = self.network.loc[i, 'rid']
            rids.append(rid)

        seg_rid = self.network.loc[current_segment, 'rid']

        for x in range(len(seg_rid)):
            if seg_rid[x] == '.':
                dot_pos = x
        length = len(seg_rid)
        segment_places = (length - 1) - dot_pos

        seg = seg_rid[-segment_places:]

        next_seg = int(seg) + 1

        chain = seg_rid[:dot_pos]

        next_rid = '{}.{}'.format(chain, next_seg)

        if next_rid in rids:
            for i in self.network.index:
                if self.network.loc[i, 'rid'] == next_rid:
                    next_id = i
        else:
            next_id = None

        if next_id is not None:
            return next_id
        else:
            return None

    def get_next_chain(self, current_segment):
        """
        Find the next chain in the network
        :param current_segment: int - identifier of chosen segment
        :return: int - identifier of first reach in next chain
        """

        rids = []
        for i in self.network.index:
            rid = self.network.loc[i, 'rid']
            rids.append(rid)

        seg_rid = self.network.loc[current_segment, 'rid']

        for x in range(len(seg_rid)):
            if seg_rid[x] == '.':
                dot_pos = x

        chain = seg_rid[:dot_pos]
        next_chain = '{}.{}'.format(int(chain) + 1, 1)

        if next_chain in rids:
            for i in self.network.index:
                if self.network.loc[i, 'rid'] == next_chain:
                    next_id = i
        else:
            next_id = None

        if next_id is not None:
            return next_id
        else:
            return None

    def seg_id_from_rid(self, rid):
        """

        :param rid:
        :return:
        """

        for i in self.network.index:
            if self.network.loc[i, 'rid'] == rid:
                seg = i

        return seg
