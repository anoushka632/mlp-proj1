pacman::p_load(tidyverse)
dat <- read_csv("the_office.csv")

dat %>% select_if(is.character)

ddat <- dat %>% mutate(across(where(is.character), strsplit, split = ";")) %>% glimpse()
x <- "1;2;3;4;5"

ddat %>% 
  group_by(writer) %>% 
  summarise(avg_rating = mean(imdb_rating),
            total_votes = sum(total_votes),
            total_n     = n()) %>% arrange(desc(avg_rating)) %>% head()

            
ddat %>% unnest(main_chars) %>%
  group_by(main_chars) %>% 
  filter(n() > 150) %>%
  ggplot() +
  aes(x = main_chars, y = imdb_rating, colour = main_chars) +
  geom_jitter() +
  theme_bw()

ddat %>% ggplot() +
  aes(x=as.factor(season), y = imdb_rating) + 
  geom_boxplot(width = .5) + 
  geom_jitter(aes(color = total_votes), width = .15, size = 1.75)+
  theme_bw() +
  scale_color_gradient2(high = "red", mid = "orange", midpoint = median(ddat$total_votes)) +
  labs(y = "imdb rating", x = "season")

ddat %>% 
  mutate(season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode",season:episode, sep = "-") %>%
  mutate(season_episode = as_factor(season_episode)) %>% 
  ggplot() +
  aes(x = season_episode, y = imdb_rating) +
  geom_point()

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
  summarise(episode = floor(n()/2)) %>% 
  mutate(labels = paste0("S", season),
         season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode", season:episode, sep = "-")

pdat <- ddat %>% 
  mutate(season = paste0("S", season),
         episode = paste0("EP", episode)) %>% 
  unite("season_episode",season:episode, sep = "-") %>%
  left_join(label_dat) %>% 
  mutate(season_episode = fct_relevel(as_factor(season_episode), ep_levels)) 

pdat %>% 
  ggplot() +
  aes(x = season_episode, y = imdb_rating) +
  geom_point()  +
  geom_line(aes(group = 1)) +
  geom_vline(xintercept = vline_mark$episode, linetype = 'dashed') +
  geom_text(aes(y = 6, label = labels), colour = 'black', size = 8) +
  scale_x_discrete(breaks = season_brks) +
  ylim(c(4.5,10)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = -45, hjust = -.05))
