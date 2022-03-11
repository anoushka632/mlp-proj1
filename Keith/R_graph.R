pacman::p_load(tidyverse)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dat <- read_csv("../the_office.csv")
ddat <- dat %>% mutate(across(where(is.character), strsplit, split = ";"))
x <- "1;2;3;4;5"

ep_levels <- ddat %>% 
  mutate(season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode",season:episode, sep = "-") %>%
  mutate(season_episode = season_episode) %>% 
  {.$season_episode}

season_brks <- ep_levels %>% 
  {.[seq(1, length(.),5)]}

vline_mark <- ddat %>% 
  group_by(season) %>% 
  summarise(episode = n()) %>% 
  mutate(episode = cumsum(episode)+ 1)
 
label_dat <- ddat %>% 
  group_by(season) %>% 
  summarise(episode = episode[(n()%/%2)]) %>% 
  mutate(labels = paste0("S", season),
         season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode", season:episode, sep = "-")

pdat <- ddat %>% 
  mutate(season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode",season:episode, sep = "-") %>%
  left_join(label_dat) %>% 
  mutate(season_episode = fct_relevel(as_factor(season_episode), ep_levels)) %>% 
  select(season_episode, imdb_rating, labels)

pdat %>%  
  ggplot() +
  aes(x = season_episode, y = imdb_rating) +
  geom_point(shape = 1)  +
  geom_line(aes(group = 1)) +
  geom_vline(xintercept = vline_mark$episode, linetype = 'dashed') +
  geom_text(aes(y = 9.5, label = labels), colour = 'black', size = 6) +
  scale_x_discrete(breaks = season_brks) +
  ylim(c(6.5,10)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = -45, hjust = -.05))

