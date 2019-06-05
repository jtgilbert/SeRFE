# imports
import geopandas as gpd
import pandas as pd
import numpy as np
import network_topology as nt


class SerfeModel:
    """
    This class runs the dynamic sediment balance model
    """

    def __init__(self, hydrograph, flow_exp, network, mannings_n=0.4, tl_factor=15):
        """

        :param hydrograph: csv file filled in with hydrograph information
        :param flow_exp: discharge-drainage area relationship exponent (can be found in plot produced from hydrology tool)
        :param network: drainage network shapefile
        :param mannings_n: a value for an average Manning's n for the basin
        :param tl_factor: a total load factor to convert bedload transport capacity into total load transport capacity
        """

        self.hydrographs = pd.read_csv(hydrograph, index_col='Gage')
        self.flow_exp = flow_exp  # need the b in the equation for flow~DA so that you can recalculate a at each time step
        self.network = gpd.read_file(network)
        self.mannings_n = mannings_n
        self.tl_factor = tl_factor
        self.streams = network

        self.nt = nt.TopologyTools(network)

        # subset of hydrographs above dams
        segs_ab_dams = []
        for x in self.hydrographs.index:
            print x
            if round(self.network.loc[self.hydrographs.loc[x, 'segid'], 'Drain_Area']) == round(self.network.loc[self.hydrographs.loc[x, 'segid'], 'eff_DA']):
                segs_ab_dams.append(self.hydrographs.loc[x, 'segid'])

        self.hyd_ab_dams = self.hydrographs[self.hydrographs['segid'].isin(segs_ab_dams)]

        # obtain number of time steps for output table
        time = np.arange(1, self.hydrographs.shape[1]-3, 1, dtype=np.int)

        # build multi-index dataframe for storing data/outputs
        self.network = gpd.read_file(network)
        segments = np.arange(0, len(self.network.index + 1), 1)

        ydim = len(time)*len(segments)
        zeros = np.zeros((ydim, 7))

        iterables = [time, segments]
        index = pd.MultiIndex.from_product(iterables, names=['time', 'segment'])

        self.outdf = pd.DataFrame(zeros, index=index, columns=['Q', 'Qs', 'Qs_out', 'CSR', 'Store_chan', 'Store_tot', 'Store_delta'])  # add the names of attributes

    def find_flow_coef(self, Q, DA):
        """
        finds the coefficient in the drainage area - discharge relationship to extrapolate flow values
        :param Q: the discharge for a given time step
        :param DA: the drainage area of the location associated with the discharge
        :return: a coeffiecient for the drainage area - discharge relationship
        """
        a = Q / DA**self.flow_exp

        return a

    def find_nearest_gage(self, segid, above_dams=False):
        """
        Finds the nearest gage to a given stream segment
        :param segid: segment ID
        :param above_dams: boolean - is the gage above all dams?
        :return: the ID of the nearest gage
        """
        seg_geom = self.network.loc[segid, 'geometry']
        x_coord = seg_geom.boundary[0].xy[0][0]
        y_coord = seg_geom.boundary[0].xy[1][0]

        if above_dams is False:
            dist = []
            for x in self.hydrographs.index:
                d = np.sqrt((x_coord-self.hydrographs.loc[x, 'Easting'])**2 + (y_coord-self.hydrographs.loc[x, 'Northing'])**2)
                dist.append(d)
            min_dist = min(dist)
            for z in range(len(dist)):
                if dist[z] == min_dist:
                    gage = self.hydrographs.index[z]
        else:
            dist = []
            for y in self.hyd_ab_dams.index:
                d = np.sqrt((x_coord-self.hyd_ab_dams.loc[y, 'Easting'])**2 + (y_coord-self.hyd_ab_dams.loc[y, 'Northing'])**2)
                dist.append(d)
            min_dist = np.min(dist)
            for z in range(len(dist)):
                if dist[z] == min_dist:
                    gage = self.hyd_ab_dams.index[z]

        return gage

    def get_upstream_qs(self, time, segid):
        """
        obtains the sediment flux from the adjacent upstream segment(s)
        :param time: time step
        :param segid: segment ID
        :return: sediment flux (tonnes)
        """
        us_seg = self.nt.find_us_seg(segid)
        us_seg2 = self.nt.find_us_seg2(segid)

        if us_seg is not None:
            usqs1 = self.outdf.loc[(time, us_seg), 'Qs_out']
        else:
            usqs1 = 0.

        if us_seg2 is not None:
            usqs2 = self.outdf.loc[(time, us_seg2), 'Qs_out']
        else:
            usqs2 = 0.

        usqs_tot = usqs1 + usqs2

        return usqs_tot

    def get_direct_qs(self, segid):
        """
        obtains the hillslope sediment delivery
        :param segid: segment ID
        :return: sediment flux (tonnes)
        """
        # assume sediment bulk density
        sed_density = 2.6  # tonne/m^3; should this be reduced to represent bulk density...?

        # calculate delivery from denudation rate
        hillslope_da = self.network.loc[segid, 'direct_DA'] - (self.network.loc[segid, 'fp_area']/1000000.)
        vol_sed = ((self.network.loc[segid, 'denude']/1000.)/365.)*(hillslope_da*1000000.)  # m^3; assumes daily time step
        dir_qs = vol_sed * sed_density

        return dir_qs

    def transport_capacity(self, Q, f_sand, w, S, D):  # denudation is all sediment whereas transport is just bedload. Just use proportion (e.g. bl = 25% total..?) and sep gravel and sand..?
        """
        calculates bedload transport capacity using Lammers and Bledsoe 2018
        :param Q: flow (cms)
        :param f_sand: fraction of sand in the bed (0-1)
        :param w: channel width
        :param S: bed slope
        :param D: median grain size
        :return: bedload transport capacity (tonnes)
        """
        # variables
        rho = 1000.
        rho_s = 2650.
        g = 9.8
        om_crit_s = 0.32*f_sand**2 - 0.24*f_sand + 0.095
        om_crit = om_crit_s * rho * (((rho_s / rho) - 1) * g * (D / 1000.)) ** (3. / 2.)
        om_crit_s_sand = 0.08
        om_crit_sand = om_crit_s_sand * rho * (((rho_s / rho) - 1) * g * (0.1 / 1000.)) ** (3. / 2.)  # D50 of sand?? 0.1

        # determine if stream power exceeds critical threshold
        om = (rho * g * Q * S) / w
        if om >= om_crit:
            rate_gravel = 0.000143 * (om - om_crit) ** (3. / 2.) * D ** (-0.5) * (Q / w) ** (-0.5)  # Lammers et al bedload equation (tonnes)
            cap_gravel = rate_gravel * w * 86400. / 1000.  # convert to tonnes/day
        else:
            cap_gravel = 0.

        if om >= om_crit_sand:
            rate_sand = 0.000143 * (om - om_crit_sand) ** (3. / 2.) * 0.1 ** (-0.5) * (Q / w) ** (-0.5)  # assumes sand D = 0.1
            cap_sand = rate_sand * w * 86400. / 1000.
        else:
            cap_sand = 0

        cap_bl = cap_gravel + cap_sand

        # if om >= om_crit_tot:
        #     rate_tot = 0.0214 * (om - om_crit_tot) ** (3. / 2.) * 0.001 ** (-1) * (Q / w) ** (-5. / 6.)  # Lammers et al total load equation (tonnes)
        #     cap_tot = (Q * 86400) * (rate_tot / 1000000.) * 2.6  # convert ppm to tonnes/day
        # else:
        #     cap_tot = 0.

        return cap_bl, cap_gravel, cap_sand

    def apply_to_reach(self, segid, time, gage):
        """
        applies the SeRFE logic to a given reach
        :param segid: segment ID
        :param time: time step
        :param gage: gage ID
        :return:
        """
        # above dams, simple flow calc
        if self.network.loc[segid, 'eff_DA'] == self.network.loc[segid, 'Drain_Area']:
            flow_coef = self.find_flow_coef(self.hydrographs.loc[gage, str(time)], self.hydrographs.loc[gage, 'DA'])
            flow = flow_coef * self.network.loc[segid, 'Drain_Area']**self.flow_exp
        # below dams find gage above dams for equn then add dam flow.
        else:  # segid['eff_DA'] != segid['Drain_Area']:
            q_t = self.hydrographs.loc[gage, str(time)]
            gage2 = self.find_nearest_gage(segid, above_dams=True)
            flow_coef = self.find_flow_coef(self.hyd_ab_dams.loc[gage2, str(time)], self.hyd_ab_dams.loc[gage2, 'DA'])
            flow = (flow_coef * self.network.loc[segid, 'eff_DA']**self.flow_exp) + q_t

        # set flow threshold; < 0.2 cfs goes to 0
        if flow < 0.0055:
            flow = 0.

        # mannings n effect on depth
        depth = ((self.mannings_n * flow) / (((self.network.loc[segid, 'w_low']+self.network.loc[segid, 'w_bf'])/2)*self.network.loc[segid, 'Slope']**0.5))**0.6

        # find upstream qs input
        qs_us = self.get_upstream_qs(time, segid)

        # find direct qs input (hillslopes)
        qs_dir = self.get_direct_qs(segid)  # tonnes

        if self.network.loc[segid, 'fp_area'] != 0.:
            qs_channel = qs_dir*self.network.loc[segid, 'confine']
            qs_fp = qs_dir - qs_channel  # tonnes
            fp_store = ((self.network.loc[segid, 'fp_area']*self.network.loc[segid, 'fp_thick'])*1.) + qs_fp  # bulk density 1 tonne/m3
            delta_fp_thick = qs_fp / self.network.loc[segid, 'fp_area']  # bulk density 1 tonne/m3
            fp_thick = self.network.loc[segid, 'fp_thick'] + delta_fp_thick  # meters
        else:
            qs_channel = qs_dir
            fp_store = 0.
            fp_thick = 0.

        # find transport capacity
        f_sand = self.network.loc[segid, 'f_sand']
        if flow >= 1.5 * self.network.loc[segid, 'Q2 (cms)']:
            w = self.network.loc[segid, 'w_flood']
        elif flow >= self.network.loc[segid, 'Q2 (cms)'] and flow < (1.5 * self.network.loc[segid, 'Q2 (cms)']):
            w = (self.network.loc[segid, 'w_bf'] + self.network.loc[segid, 'w_flood'])/2
        elif flow >= (0.5 * self.network.loc[segid, 'Q2 (cms)']) and flow < self.network.loc[segid, 'Q2 (cms)']:
            w = self.network.loc[segid, 'w_bf']
        elif flow >= (0.1 * self.network.loc[segid, 'Q2 (cms)']) and flow < (0.5 * self.network.loc[segid, 'Q2 (cms)']):
            w = (self.network.loc[segid, 'w_bf'] + self.network.loc[segid, 'w_low'])/2
        else:  # flow < (0.05 * self.network.loc[segid, 'Q2 (cms)']):
            w = self.network.loc[segid, 'w_low']
        S = self.network.loc[segid, 'Slope']
        D = self.network.loc[segid, 'D_pred']
        cap_bl, cap_gravel, cap_sand = self.transport_capacity(flow, f_sand, w, S, D)
        cap_tot = cap_bl * self.tl_factor  # this is very uncertain.... just estimated from Williams 1979

        # apply transport/routing logic

        # mig_rate = 0.00329 * ((1000*9.81*flow*S)/w)  # from inital analysis
        sp_crit = (9810 * self.network.loc[segid, 'Qc'] * S) / self.network.loc[segid, 'w_bf']
        excess_sp = float(((1000*9.81*flow*S)/self.network.loc[segid, 'w_bf']) - sp_crit)
        if excess_sp <= 0:
            excess_sp = 0.00000001
        mig_rate = 0.016 * excess_sp**0.9
        # mig_rate = exp(0.00078*((1000*9.81*flow*S)/w))-1
        if mig_rate < 0.:
            mig_rate = 0.

        if time == 1:
            prev_ch_store = 0.
        else:
            prev_ch_store = self.outdf.loc[(time-1, segid), 'Store_chan']

        if cap_tot < (qs_channel + qs_us + prev_ch_store):
            qs_out = cap_tot
            if self.network.loc[segid, 'confine'] != 1.:
                if depth < self.network.loc[segid, 'fp_thick']:
                    channel_store = (qs_channel + qs_us + prev_ch_store) - qs_out
                    delta_h = ((channel_store-prev_ch_store)*1) / (0.5*self.network.loc[segid, 'w_bf']*self.network.loc[segid, 'Length_m'])
                    self.network.loc[segid, 'Slope'] = self.network.loc[segid, 'Slope'] + (delta_h/self.network.loc[segid, 'Length_m'])
                else:
                    channel_ratio = (self.network.loc[segid, 'Length_m']*self.network.loc[segid, 'w_bf'])/((self.network.loc[segid, 'Length_m']*self.network.loc[segid, 'w_bf'])+self.network.loc[segid, 'fp_area'])
                    tl_remain = (qs_channel + qs_us + prev_ch_store) - qs_out
                    bl_remain = tl_remain / self.tl_factor
                    wash_remain = tl_remain - bl_remain
                    channel_store = bl_remain + (wash_remain*channel_ratio)  # remaining bedload and portion of washload overlaying floodplain
                    delta_h = ((channel_store - prev_ch_store) * 1) / (0.5 * self.network.loc[segid, 'w_bf'] * self.network.loc[segid, 'Length_m'])
                    self.network.loc[segid, 'Slope'] = self.network.loc[segid, 'Slope'] + (delta_h/self.network.loc[segid, 'Length_m'])
                    fp_store = fp_store + (wash_remain*(1-channel_ratio))  # add to fp storage from hillslope contribution
                    fp_thick = fp_store / self.network.loc[segid, 'fp_area']
            else:
                channel_store = (qs_channel + qs_us + prev_ch_store) - qs_out
                delta_h = ((channel_store - prev_ch_store) * 1) / (0.5 * self.network.loc[segid, 'w_bf'] * self.network.loc[segid, 'Length_m'])
                self.network.loc[segid, 'Slope'] = self.network.loc[segid, 'Slope'] + (delta_h/self.network.loc[segid, 'Length_m'])

            csr = cap_tot / (qs_channel + qs_us + prev_ch_store)

        elif cap_tot > (qs_channel + qs_us + prev_ch_store):
            if self.network.loc[segid, 'confine'] != 1.:
                fp_recr = mig_rate * (self.network.loc[segid, 'Length_m'] * (1 - self.network.loc[segid, 'confine'])) * fp_thick  # bulk density 1 tonne/m3
                qs_out = (qs_channel + qs_us + prev_ch_store) + fp_recr
                fp_store = fp_store - fp_recr
                channel_store = 0.
                delta_h = ((channel_store - prev_ch_store) * 1) / (0.5 * self.network.loc[segid, 'w_bf'] * self.network.loc[segid, 'Length_m'])
                self.network.loc[segid, 'Slope'] = self.network.loc[segid, 'Slope'] + (delta_h/self.network.loc[segid, 'Length_m'])
                # update fp area...?  if i hold width constant that the change is storage should be represented by lower fp surface (see how line 166 elevates it.
                fp_recr_thick = fp_recr / self.network.loc[segid, 'fp_area']  # bulk density 1 tonne / m3
                fp_thick = fp_thick - fp_recr_thick

                csr = cap_tot / (qs_channel + qs_us + prev_ch_store + fp_recr)
            else:
                channel_store = 0.
                delta_h = ((channel_store - prev_ch_store) * 1) / (0.5 * self.network.loc[segid, 'w_bf'] * self.network.loc[segid, 'Length_m'])
                self.network.loc[segid, 'Slope'] = self.network.loc[segid, 'Slope'] + (delta_h/self.network.loc[segid, 'Length_m'])
                qs_out = (qs_channel + qs_us + prev_ch_store)
                csr = cap_tot / (qs_channel + qs_us + prev_ch_store)

        else:  # if in == out
            qs_out = qs_channel + qs_us + prev_ch_store
            channel_store = 0.
            csr = 1.

        # if you are at a dam, qs_out = 0
        next_reach = self.nt.get_next_reach(segid)
        if next_reach is not None:
            if self.network.loc[next_reach, 'eff_DA'] < self.network.loc[segid, 'eff_DA']:
                qs_out = 0.

        store_tot = channel_store + fp_store

        if self.network.loc[segid, 'confine'] != 1.:
            self.network.loc[segid, 'fp_thick'] = fp_thick

        # update output table
        self.outdf.loc[(time, segid), 'Q'] = flow
        self.outdf.loc[(time, segid), 'Qs'] = qs_channel + qs_us + prev_ch_store
        self.outdf.loc[(time, segid), 'Qs_out'] = qs_out
        self.outdf.loc[(time, segid), 'CSR'] = csr
        self.outdf.loc[(time, segid), 'Store_tot'] = store_tot
        self.outdf.loc[(time, segid), 'Store_chan'] = channel_store
        if time > 1:
            self.outdf.loc[(time, segid), 'Store_delta'] = store_tot - (self.outdf.loc[(time-1, segid), 'Store_tot'])
        else:
            self.outdf.loc[(time, segid), 'Store_delta'] = 0

        return

    def run_first_order(self, time):
        """
        runs the model for all first order stream segments
        :param time: time step
        :return:
        """
        seg = self.nt.seg_id_from_rid('1.1')
        time = time
        gage = self.find_nearest_gage(seg)

        while seg is not None:
            self.apply_to_reach(seg, time, gage)

            next_reach = self.nt.get_next_reach(seg)

            if next_reach is not None:
                if self.network.loc[next_reach, 'confluence'] == 0:
                    if self.network.loc[next_reach, 'eff_DA'] < self.network.loc[seg, 'eff_DA']:
                        gage = self.find_nearest_gage(next_reach)  # dam impacted
                    else:
                        pass
                else:
                    next_reach = self.nt.get_next_chain(next_reach)
            else:
                next_reach = self.nt.get_next_chain(seg)

            seg = next_reach

        return

    def run_below_confluences(self, time):
        """
        runs the model for all stream segments below confluences (greater than first order)
        :param time: time step
        :return:
        """
        conf_list = []  # is there a way to get rid of this type of for loop
        da_vals = []
        for i in self.network.index:
            if self.network.loc[i, 'confluence'] == 1:
                conf_list.append(i)
                da_vals.append(self.network.loc[i, 'Drain_Area'])
        sort = np.argsort(da_vals)
        conf_list = [conf_list[i] for i in sort]

        # the sorting by da alone isn't working. Need to set qs_out equal to some nodata val and then make sure there's
        # a value for both upstream segments before running the model on the segment

        while len(conf_list) > 0:
            for x in conf_list:
                seg = x
                us = self.nt.find_us_seg(seg)
                us2 = self.nt.find_us_seg2(seg)

                if self.outdf.loc[(time, us), 'Qs_out'] == -9999 or self.outdf.loc[(time, us2), 'Qs_out'] == -9999:
                    pass
                else:
                    time = time
                    hyd = self.find_nearest_gage(seg)  # may change this to just be the same as its upstream segment... (stored in outdf)

                    while seg is not None:
                        self.apply_to_reach(seg, time, hyd)

                        next_reach = self.nt.get_next_reach(seg)

                        if next_reach is not None:
                            if self.network.loc[next_reach, 'confluence'] == 0:
                                if self.network.loc[next_reach, 'eff_DA'] < self.network.loc[seg, 'eff_DA']:
                                    hyd = self.find_nearest_gage(next_reach)
                                else:
                                    pass
                            else:
                                next_reach = None
                        else:
                            pass

                        seg = next_reach

                    conf_list.remove(x)

        return

    def run_model(self, spinup=False):
        """
        method that runs the dynamic SeRFE model
        :param spinup: boolean - True of running spinup period, False if saving outputs
        :return: a dataframe with two index columns (Segment ID and time step) containing model outputs for each segment
        """
        total_t = self.hydrographs.shape[1]-5
        time = 1

        while time <= total_t:
            print 'day ' + str(time)

            # set qs_out initially to -9999
            for i in range(len(self.outdf.index.levels[1])):
                self.outdf.loc[(time, i), 'Qs_out'] = -9999

            # apply denudation rate to each segment
            for i in self.network.index:
                if self.network.loc[i, 'dist_start'] != -9999:
                    if time in range(self.network.loc[i, 'dist_start'], (self.network.loc[i, 'dist_end']+1)):
                        self.network.loc[i, 'denude'] = np.random.gamma(self.network.loc[i, 'dist_g_sh'], self.network.loc[i, 'dist_g_sc'])
                    else:
                        self.network.loc[i, 'denude'] = np.random.gamma(self.network.loc[i, 'g_shape'], self.network.loc[i, 'g_scale'])
                else:
                    self.network.loc[i, 'denude'] = np.random.gamma(self.network.loc[i, 'g_shape'], self.network.loc[i, 'g_scale'])

            # run the model for given time step
            print 'running first order'
            self.run_first_order(time)

            print 'running below confluences'
            self.run_below_confluences(time)

            # reset denude rates to -9999, do I need to do this or will it just overwrite?

        if spinup:
            self.network.to_file(self.streams)

            return None
        else:

            return self.outdf
