// assets
import { PieChartOutlined, DashboardOutlined, DollarOutlined, FundProjectionScreenOutlined } from '@ant-design/icons';

// icons
const icons = {
  PieChartOutlined,
  DashboardOutlined,
  DollarOutlined,
  FundProjectionScreenOutlined
};

// ==============================|| MENU ITEMS - DASHBOARD ||============================== //

const dashboard = {
  id: 'group-dashboard',
  title: 'Navigation',
  type: 'group',
  children: [
    {
      id: 'portfolio',
      title: 'Portfolio',
      type: 'item',
      url: '/portfolio',
      icon: icons.PieChartOutlined
    },
    {
      id: 'dashboard',
      title: 'Macro & Industry',
      type: 'item',
      url: '/dashboard',
      icon: icons.DashboardOutlined,
      breadcrumbs: false
    },
    {
      id: 'companies',
      title: 'Companies',
      type: 'item',
      url: '/companies',
      icon: icons.DollarOutlined
    },
    {
      id: 'action',
      title: 'Actionable Suggestions',
      type: 'item',
      url: '/action',
      icon: icons.FundProjectionScreenOutlined
    }
  ]
};

export default dashboard;
